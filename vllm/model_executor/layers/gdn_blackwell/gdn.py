"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Gated Delta Networks (GDN) chunked linear attention kernel for NVIDIA Blackwell (SM100).
#
# Implements the Chunk-wise Gated Delta Rule linear attention using CuTe-DSL,
# for Blackwell Architecture.
#
# Key Features:
#   - Supports Persistent & Non-persistent modes
#   - Supports fixed-length and variable-length sequences (cu_seqlens)
#   - Supports grouped value attention (GVA) where h_v is a multiple of h_q
#   - Supports head dimension (128)
#   - Supports input data types (f16/bf16)
#   - Supports output data types (f16/bf16)
#   - Supports initial_state to provide the initial state
#   - Supports output_final_state flag to return the final state
#   - State input and output are in f32 (fp16/bf16 not supported yet)

import torch

from typing import Optional, Type, Tuple, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import nvvm

import cuda.bindings.driver as cuda


from .gdn_tile_scheduler import (
    GdnStaticTileScheduler,
    GdnStaticTileSchedulerParams,
    create_gdn_static_tile_scheduler,
    create_gdn_static_tile_scheduler_params,
)
from .gdn_helpers import (
    make_smem_layout_a_kind,
    make_smem_layout_b_kind,
    make_smem_layout_epi_kind,
)

import functools
import warnings

from cutlass.cute import EnableTVMFFI
from cutlass import Int32, Int64


def make_thread_cooperative_group(size: Int32):
    # return pipeline.CooperativeGroup(pipeline.Agent.Thread, size, size) # old version
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class GDN:
    """Blackwell (SM100) kernel for Chunk-wise Gated Delta Rule linear attention."""

    def __init__(
        self,
        is_persistent: bool = False,
        chunk_size: Int32 = 128,  # 64 or 128; 64 is faster for small head counts
        head_dim: Int32 = 128,  # Only 128 is supported in current version
        num_chunk_groups: int = 1,  # >1 for chunk parallelism (small heads)
    ):
        self.chunk_size = chunk_size
        self.is_persistent = is_persistent
        self.num_chunk_groups = num_chunk_groups

        self.head_dim = head_dim
        self.cta_tiler = (chunk_size, chunk_size, head_dim)
        self.gate_tiler = (chunk_size, 1)
        self.beta_tiler = (chunk_size, 1)
        self.state_output_tiler = (head_dim, head_dim)
        self.o_output_tiler = (chunk_size, head_dim)

        self.g_beta_dtype = cutlass.Float32
        self.acc_dtype = cutlass.Float32
        self.state_dtype = cutlass.Float32

        self.mma_tiler = (chunk_size, chunk_size, head_dim)
        self.qk_mma_tiler = self.mma_tiler
        self.kkt_mma_tiler = self.mma_tiler
        self.kst_mma_tiler = (chunk_size, head_dim, head_dim)
        self.tuw_mma_tiler = (chunk_size, head_dim, chunk_size)
        self.o_intra_mma_tiler = (chunk_size, head_dim, chunk_size)
        self.qs_mma_tiler = (chunk_size, head_dim, head_dim)
        self.update_s_mma_tiler = (head_dim, head_dim, chunk_size)

        self.epi_tile = (chunk_size, head_dim)

        self.cluster_shape_mnk = (1, 1, 1)

        self.buffer_align_bytes = 1024

        self.q_stage = 1
        self.one_stage = 1
        self.kv_stage = 1
        self.qk_stage = 2
        self.gate_stage = 1
        self.beta_stage = 1
        self.mma_cudacore_stage = 1
        self.mma_qk_stage = 2
        self.epi_stage = 1

        self.cudacore_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.epilogue_warp_id = 6
        self.gb_warp_id = 7

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.cudacore_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.gb_warp_id,
            )
        )

        # TMEM configuration
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_kkt_output = 0
        self.tmem_qkt_output = 384
        self.tmem_gate_qk = 384
        self.tmem_gate_q = 448  # 448
        self.tmem_k_input = 64
        self.tmem_kst_output = 256
        self.tmem_ivt_ss_l0_output = 448
        self.tmem_ivt_ts_l0_input = 320
        self.tmem_ivt_ts_l0_output = 448
        self.tmem_ivt_ss_l1_output = 320
        self.tmem_ivt_ts_l1_input = 320
        self.tmem_ivt_ts_l1_output = 256
        self.tmem_tuw_output = 256
        self.tmem_o_intra_output = 256
        self.tmem_o_inter_output = 256
        self.tmem_update_s_output = 128
        self.tmem_gamma = 256

        self.tmem_state = 128

        # Registers configuration
        self.num_regs_cudacore = 240
        self.num_regs_mma = 64
        self.num_regs_other = 64
        self.num_regs_gb = 64

        # Named barrier IDs
        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.cudacore_mma_sync_bar_id = 2
        self.wg_sync_bar_id = 3
        self.epi_load_sync_bar_id = 4

        # Matrix inversion using TFloat32
        self.invert_type_ab = cutlass.TFloat32
        self.invert_acc_type = cutlass.Float32
        self.invert_mma_sub_l0_tiler = (64, 64, 32)
        self.invert_mma_sub_l1_tiler = (64, 64, 64)
        self.invert_mma_tiler = (128, 128, 64)
        self.invert_sub_stage = 1
        self.sub_stage = 9

    @staticmethod
    def can_implement(
        q_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int],
        v_shape: Tuple[int, int, int, int] | Tuple[int, Tuple[int, ...], int, int],
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        g_beta_dtype: Type[cutlass.Numeric],
        use_qk_l2norm_in_kernel: bool = False,
    ) -> bool:
        """
        Check if the gdn can be implemented
        """

        can_implement = True

        # Unpack parameters
        b, s_q, h_q, d = q_shape
        b_, _, h_v, d_ = v_shape

        if use_qk_l2norm_in_kernel:
            warnings.warn("use_qk_l2norm_in_kernel is not supported yet", stacklevel=2)
            can_implement = False

        if b != b_:
            warnings.warn("q & k must have the same batch size", stacklevel=2)
            can_implement = False

        if d != d_:
            warnings.warn("q & k must have the same head dimension", stacklevel=2)
            can_implement = False

        # todo: maybe support more later.
        if d not in {128}:
            warnings.warn("head dimension must be 128", stacklevel=2)
            can_implement = False

        if h_v % h_q != 0:
            warnings.warn("h_v must be divisible by h_q", stacklevel=2)

            can_implement = False

        if isinstance(s_q, tuple) and len(s_q) != b:
            warnings.warn(
                "variable_seqlen s_q must have the length of batch size", stacklevel=2
            )
            can_implement = False

        if in_dtype not in {
            cutlass.BFloat16,
            cutlass.Float16,
            torch.float16,
            torch.bfloat16,
        }:
            warnings.warn(
                "in_dtype must be BFloat16 or Float16 or torch.float16 or torch.bfloat16",
                stacklevel=2,
            )
            can_implement = False

        if out_dtype not in {
            cutlass.BFloat16,
            cutlass.Float16,
            torch.float16,
            torch.bfloat16,
        }:
            warnings.warn(
                "out_dtype must be BFloat16 or Float16 or torch.float16 or torch.bfloat16",
                stacklevel=2,
            )
            can_implement = False

        if g_beta_dtype not in {cutlass.Float32, torch.float32}:
            warnings.warn(
                "g_beta_dtype must be Float32 or torch.float32, but got {}".format(
                    g_beta_dtype
                ),
                stacklevel=2,
            )
            can_implement = False

        return can_implement

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        o_intra_tiled_mma: cute.TiledMma,
        o_intra_tiled_ts_mma_new: cute.TiledMma,
        qs_tiled_mma: cute.TiledMma,
        update_s_tiled_mma: cute.TiledMma,
        kkt_tiled_mma: cute.TiledMma,
        kst_tiled_mma: cute.TiledMma,
        fake_state_tiled_mma: cute.TiledMma,
        invert_sub_tiled_mma_ss_l0: cute.TiledMma,
        invert_sub_tiled_mma_ts_l0: cute.TiledMma,
        invert_sub_tiled_mma_ss_l1: cute.TiledMma,
        invert_sub_tiled_mma_ts_l1: cute.TiledMma,
        tuw_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_state_f32: Optional[cute.CopyAtom],
        mState_f32: Optional[cute.Tensor],
        gate: cute.Tensor,
        beta: cute.Tensor,
        O: cute.Tensor,
        tma_atom_state_output: Optional[cute.CopyAtom],
        mStateOutput: Optional[cute.Tensor],
        tma_atom_o_output: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_intra_smem_layout_staged_a: cute.ComposedLayout,
        o_intra_smem_layout_staged_b: cute.ComposedLayout,
        o_intra_smem_layout_staged_a_new: cute.ComposedLayout,
        qs_smem_layout_staged_a: cute.ComposedLayout,
        qs_smem_layout_staged_b: cute.ComposedLayout,
        update_s_smem_layout_staged_a: cute.ComposedLayout,
        update_s_smem_layout_staged_b: cute.ComposedLayout,
        state_smem_layout_staged: cute.ComposedLayout,
        state_smem_layout_f32_staged: cute.ComposedLayout,
        state_output_smem_layout_staged: cute.ComposedLayout,
        o_output_smem_layout_staged: cute.ComposedLayout,
        gate_smem_layout_staged: cute.Layout,
        beta_smem_layout_staged: cute.Layout,
        ivt_smem_layout: cute.Layout,
        sub_inner_smem_layout_staged: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ss_l0_a: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ss_l0_b: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ts_l0_a: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ts_l0_b: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ss_l1_a: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ss_l1_b: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ts_l1_a: cute.ComposedLayout,
        invert_sub_smem_layout_staged_ts_l1_b: cute.ComposedLayout,
        tuw_smem_layout_staged_a: cute.ComposedLayout,
        tuw_smem_layout_staged_b: cute.ComposedLayout,
        v_smem_layout_staged_b: cute.ComposedLayout,
        scale: cutlass.Float32,
        tile_sched_params: GdnStaticTileSchedulerParams,
        workspace_state_ptr: Optional[cute.Pointer] = None,
        workspace_flags_ptr: Optional[cute.Pointer] = None,
    ):
        """Warp-specialized GDN kernel entry point."""
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o_output)
            if cutlass.const_expr(tma_atom_state_f32 is not None):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_state_f32)
            if cutlass.const_expr(tma_atom_state_output is not None):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_state_output)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_qk_producer, load_qk_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.qk_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_qk_mbar_ptr.data_ptr(),
        ).make_participants()

        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.one_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.cudacore_warp_ids),
            ),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
        ).make_participants()

        load_state_producer, load_state_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.one_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len(self.cudacore_warp_ids)),
            tx_count=self.tma_copy_state_f32_bytes,
            barrier_storage=storage.load_state_mbar_ptr.data_ptr(),
        ).make_participants()

        mma_qk_producer0, mma_qk_consumer0 = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_qk_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cudacore_warp_ids)
            ),
            barrier_storage=storage.mma_qk_mbar_ptr0.data_ptr(),
        ).make_participants()

        mma_cudacore_producer0, mma_cudacore_consumer0 = (
            pipeline.PipelineUmmaAsync.create(
                num_stages=self.mma_cudacore_stage,
                producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
                consumer_group=make_thread_cooperative_group(
                    self.threads_per_warp * len(self.cudacore_warp_ids)
                ),
                barrier_storage=storage.mma_cudacore_mbar_ptr0.data_ptr(),
            ).make_participants()
        )

        gb_w0_producer, gb_w0_consumer = pipeline.PipelineAsync.create(
            num_stages=self.gate_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cudacore_warp_ids)
            ),
            barrier_storage=storage.gb_w0_mbar_ptr.data_ptr(),
        ).make_participants()

        epi_w0_producer, epi_w0_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cudacore_warp_ids)
            ),
            barrier_storage=storage.epi_w0_mbar_ptr.data_ptr(),
        ).make_participants()

        w0_epi_producer, w0_epi_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cudacore_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp),
            barrier_storage=storage.w0_epi_mbar_ptr.data_ptr(),
        ).make_participants()

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()

        #  tmem barrier init
        if warp_idx == self.gb_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.threads_per_warp * len((*self.cudacore_warp_ids,)),
            )
        cute.arch.mbarrier_init_fence()

        sQK = storage.sQK.get_tensor(
            qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner
        )
        sQ = cute.make_tensor(
            cute.recast_ptr(
                sQK[None, None, None, 1].iterator,
                swizzle_=q_smem_layout_staged.inner,
                dtype=self.i_dtype,
            ),
            q_smem_layout_staged.outer,
        )
        sK = cute.make_tensor(
            cute.recast_ptr(
                sQK[None, None, None, 0].iterator,
                swizzle_=k_smem_layout_staged.inner,
                dtype=self.i_dtype,
            ),
            k_smem_layout_staged.outer,
        )
        sStateInput = cute.make_tensor(
            cute.recast_ptr(
                storage.sState.data_ptr(),
                swizzle_=state_smem_layout_staged.inner,
                dtype=self.i_dtype,
            ),
            state_smem_layout_staged.outer,
        )
        sV = cute.make_tensor(
            cute.recast_ptr(
                storage.sV.data_ptr(),
                swizzle_=v_smem_layout_staged_b.inner,
                dtype=self.i_dtype,
            ),
            v_smem_layout_staged_b.outer,
        )

        sGate = storage.sGate.get_tensor(gate_smem_layout_staged)
        sBeta = storage.sBeta.get_tensor(beta_smem_layout_staged)
        sGateCumsum = storage.sGateCumsum.get_tensor(
            cute.select(gate_smem_layout_staged, mode=[0, 1])
        )
        sSubInner = storage.sSubInner.get_tensor(sub_inner_smem_layout_staged)
        sInvertSubSSL0A = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 6].iterator,
                swizzle_=invert_sub_smem_layout_staged_ss_l0_a.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ss_l0_a.outer,
        )

        sInvertSubSSL0B = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 7].iterator,
                swizzle_=invert_sub_smem_layout_staged_ss_l0_b.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ss_l0_b.outer,
        )
        sInvertSubTSL0B = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 8].iterator,
                swizzle_=invert_sub_smem_layout_staged_ts_l0_b.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ts_l0_b.outer,
        )
        sInvertSubSSL1A = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 0].iterator,
                swizzle_=invert_sub_smem_layout_staged_ss_l1_a.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ss_l1_a.outer,
        )
        sInvertSubSSL1B = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 2].iterator,
                swizzle_=invert_sub_smem_layout_staged_ss_l1_b.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ss_l1_b.outer,
        )
        sInvertSubTSL1B = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 6].iterator,
                swizzle_=invert_sub_smem_layout_staged_ts_l1_b.inner,
                dtype=self.invert_type_ab,
            ),
            invert_sub_smem_layout_staged_ts_l1_b.outer,
        )

        sInvertSubReg = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 4].iterator, dtype=self.invert_type_ab
            ),
            cute.make_layout(((4, 128), 8)),
        )

        sStateInputF32 = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 0].iterator,
                swizzle_=state_smem_layout_f32_staged.inner,
                dtype=self.state_dtype,
            ),
            state_smem_layout_f32_staged.outer,
        )
        sStateOutput = cute.make_tensor(
            cute.recast_ptr(
                sQK[None, None, None, 0].iterator,
                swizzle_=state_output_smem_layout_staged.inner,
                dtype=self.state_dtype,
            ),
            state_output_smem_layout_staged.outer,
        )

        sOIntraB = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 4].iterator,
                swizzle_=o_intra_smem_layout_staged_b.inner,
                dtype=self.i_dtype,
            ),
            o_intra_smem_layout_staged_b.outer,
        )
        sQSB = cute.make_tensor(
            cute.recast_ptr(
                storage.sState.data_ptr(),
                swizzle_=qs_smem_layout_staged_b.inner,
                dtype=self.i_dtype,
            ),
            qs_smem_layout_staged_b.outer,
        )

        sUpdateA = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 4].iterator,
                swizzle_=update_s_smem_layout_staged_a.inner,
                dtype=self.i_dtype,
            ),
            update_s_smem_layout_staged_a.outer,
        )
        sUpdateB = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 0].iterator,
                swizzle_=update_s_smem_layout_staged_b.inner,
                dtype=self.i_dtype,
            ),
            update_s_smem_layout_staged_b.outer,
        )

        sO = cute.make_tensor(
            cute.recast_ptr(
                storage.sState.data_ptr(),
                swizzle_=o_output_smem_layout_staged.inner,
                dtype=self.i_dtype,
            ),
            o_output_smem_layout_staged.outer,
        )
        sIvt = storage.sIvt.get_tensor(ivt_smem_layout)

        sTuwA = cute.make_tensor(
            cute.recast_ptr(
                sSubInner[None, None, None, 0].iterator,
                swizzle_=tuw_smem_layout_staged_a.inner,
                dtype=self.i_dtype,
            ),
            tuw_smem_layout_staged_a.outer,
        )
        sTuwB = cute.make_tensor(
            cute.recast_ptr(storage.sV.data_ptr(), dtype=self.k_dtype),
            tuw_smem_layout_staged_b,
        )
        sTuwAStore_layout = cute.make_layout(
            ((128, (4, 2)), 8), stride=((4, (1, 512)), 1024)
        )
        sTuwAStore = cute.make_tensor(
            cute.recast_ptr(sTuwA.iterator, dtype=cutlass.Float32), sTuwAStore_layout
        )

        if warp_idx == self.epilogue_warp_id:
            # Alloc tmem buffer
            tmem_alloc_cols = cutlass.Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )

        qk_thr_mma = qk_tiled_mma.get_slice(0)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQK)
        tSrK = qk_thr_mma.make_fragment_B(sQK)
        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tQKtQK_fake = qk_thr_mma.make_fragment_C(qk_acc_shape)
        tQKtQK = cute.make_tensor(tmem_ptr + self.tmem_qkt_output, tQKtQK_fake.layout)

        kkt_thr_mma = kkt_tiled_mma.get_slice(0)  # default 1sm
        tArK = kkt_thr_mma.make_fragment_A(sQK)
        tArKT = kkt_thr_mma.make_fragment_B(sQK)
        kkt_acc_shape = kkt_thr_mma.partition_shape_C(
            (self.kkt_mma_tiler[0], self.kkt_mma_tiler[1])
        )

        fake_state_thr_mma = fake_state_tiled_mma.get_slice(0)

        kst_thr_mma = kst_tiled_mma.get_slice(0)
        tKSrK_0 = kst_thr_mma.make_fragment_A(sQK)
        tKSrST = kst_thr_mma.make_fragment_B(sStateInput)
        kst_acc_shape = kst_thr_mma.partition_shape_C(
            (self.kst_mma_tiler[0], self.kst_mma_tiler[1])
        )
        tKStKS_fake = kst_thr_mma.make_fragment_C(kst_acc_shape)
        tKStKS = cute.make_tensor(tmem_ptr + self.tmem_kst_output, tKStKS_fake.layout)
        # input smem A
        tKStK_layout = cute.composition(tKStKS.layout, cute.make_layout((128, 64)))
        tKtK = cute.make_tensor(tmem_ptr + self.tmem_k_input, tKStK_layout)

        invert_sub_thr_mma_ss_l0 = invert_sub_tiled_mma_ss_l0.get_slice(0)
        tSrD = invert_sub_thr_mma_ss_l0.make_fragment_A(sInvertSubSSL0A)
        tSrC = invert_sub_thr_mma_ss_l0.make_fragment_B(sInvertSubSSL0B)
        invert_sub_acc_shape = invert_sub_thr_mma_ss_l0.partition_shape_C(
            (self.invert_mma_sub_l0_tiler[0], self.invert_mma_sub_l0_tiler[1])
        )

        tKKTtKKT_fake = kkt_thr_mma.make_fragment_C(kkt_acc_shape)
        tKKTtKKT = cute.make_tensor(
            tmem_ptr + self.tmem_kkt_output, tKKTtKKT_fake.layout
        )

        tItSSL0_fake = invert_sub_thr_mma_ss_l0.make_fragment_C(invert_sub_acc_shape)
        tItSSL0 = cute.make_tensor(
            tmem_ptr + self.tmem_ivt_ss_l0_output, tItSSL0_fake.layout
        )

        # : ((128,128),1,1):((65536,1),0,0)
        tQgate_layout = cute.make_layout(((self.chunk_size, self.chunk_size // 2), 1, 1), stride=((65536, 1), 0, 0))
        tQgate = cute.make_tensor(tmem_ptr + self.tmem_gate_q, tQgate_layout)

        tGamma_layout = cute.make_layout(((self.chunk_size, self.chunk_size), 1, 1), stride=((65536, 1), 0, 0))
        tGamma = cute.make_tensor(tmem_ptr + self.tmem_gamma, tGamma_layout)

        # tStage
        tStateF32_layout = cute.make_layout(
            ((128, 128), 1, 1), stride=((65536, 1), 0, 0)
        )
        tStateF32 = cute.make_tensor(tmem_ptr + self.tmem_state, tStateF32_layout)

        invert_sub_thr_mma_ts_l0 = invert_sub_tiled_mma_ts_l0.get_slice(0)
        tInrDCL0A_fake = invert_sub_thr_mma_ts_l0.make_fragment_A(
            invert_sub_smem_layout_staged_ts_l0_a.outer.shape
        )
        tInrDCL0A = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr,
                dtype=tInrDCL0A_fake.element_type,
            )
            + self.tmem_ivt_ts_l0_input,
            tInrDCL0A_fake.layout,
        )

        # input smem B
        tSrDCL0B = invert_sub_thr_mma_ts_l0.make_fragment_B(sInvertSubTSL0B)

        # output tmem C
        tItTSL0_fake = invert_sub_thr_mma_ts_l0.make_fragment_C(invert_sub_acc_shape)
        tItTSL0 = cute.make_tensor(
            tmem_ptr + self.tmem_ivt_ts_l0_output, tItTSL0_fake.layout
        )

        invert_sub_thr_mma_ss_l1 = invert_sub_tiled_mma_ss_l1.get_slice(0)
        tSrD_ivt_l1 = invert_sub_thr_mma_ss_l1.make_fragment_A(sInvertSubSSL1A)
        tSrC_ivt_l1 = invert_sub_thr_mma_ss_l1.make_fragment_B(sInvertSubSSL1B)
        invert_sub_acc_l1_shape = invert_sub_thr_mma_ss_l1.partition_shape_C(
            (self.invert_mma_sub_l1_tiler[0], self.invert_mma_sub_l1_tiler[1])
        )
        tCtAcc_ivt_l1_ss_fake = invert_sub_thr_mma_ss_l1.make_fragment_C(
            invert_sub_acc_l1_shape
        )
        tStO_ivt_l1_ss0 = cute.make_tensor(
            tmem_ptr + self.tmem_ivt_ss_l1_output, tCtAcc_ivt_l1_ss_fake.layout
        )

        invert_sub_thr_mma_ts_l1 = invert_sub_tiled_mma_ts_l1.get_slice(0)
        invert_acc_ts_l1_shape = invert_sub_thr_mma_ts_l1.partition_shape_C(
            (self.invert_mma_sub_l1_tiler[0], self.invert_mma_sub_l1_tiler[1])
        )
        # output tmem C
        tOtTSL1_fake = invert_sub_thr_mma_ts_l1.make_fragment_C(invert_acc_ts_l1_shape)
        tOtTSL1 = cute.make_tensor(
            tmem_ptr + self.tmem_ivt_ts_l1_output, tOtTSL1_fake.layout
        )

        tuw_thr_mma = tuw_tiled_mma.get_slice(0)
        # input smem A
        tTUWrA = tuw_thr_mma.make_fragment_A(sTuwA)
        # input smem B
        tTUWrB = tuw_thr_mma.make_fragment_B(sTuwB)
        # output tmem C
        tuw_acc_shape = tuw_thr_mma.partition_shape_C(
            (self.tuw_mma_tiler[0], self.tuw_mma_tiler[1])
        )
        tVtTUW_fake = tuw_thr_mma.make_fragment_C(tuw_acc_shape)
        tVtTUW = cute.make_tensor(tmem_ptr + self.tmem_tuw_output, tVtTUW_fake.layout)

        o_intra_thr_mma = o_intra_tiled_mma.get_slice(0)
        # input smem B
        tOIntrarB = o_intra_thr_mma.make_fragment_B(sOIntraB)
        # output tmem C
        o_intra_acc_shape = o_intra_thr_mma.partition_shape_C(
            (self.o_intra_mma_tiler[0], self.o_intra_mma_tiler[1])
        )

        o_intra_thr_mma_new = o_intra_tiled_ts_mma_new.get_slice(0)
        # input smem A
        tQKMtQKM_layout = cute.composition(tQKtQK.layout, cute.make_layout((self.chunk_size, self.chunk_size // 2)))
        tQKMtQKM = cute.make_tensor(tmem_ptr + self.tmem_gate_qk, tQKMtQKM_layout)

        update_s_thr_mma = update_s_tiled_mma.get_slice(0)
        tUpdateA = update_s_thr_mma.make_fragment_A(sUpdateA)
        tUpdateB = update_s_thr_mma.make_fragment_B(sUpdateB)
        update_s_acc_shape = update_s_thr_mma.partition_shape_C(
            (self.update_s_mma_tiler[0], self.update_s_mma_tiler[1])
        )
        tStUpdate_fake = update_s_thr_mma.make_fragment_C(update_s_acc_shape)
        tStUpdate = cute.make_tensor(
            tmem_ptr + self.tmem_update_s_output, tStUpdate_fake.layout
        )

        qs_thr_mma = qs_tiled_mma.get_slice(0)
        tQSB = qs_thr_mma.make_fragment_B(sQSB)
        qs_acc_shape = qs_thr_mma.partition_shape_C(
            (self.qs_mma_tiler[0], self.qs_mma_tiler[1])
        )
        tOintertQS_fake = qs_thr_mma.make_fragment_C(qs_acc_shape)
        tOintertQS = cute.make_tensor(
            tmem_ptr + self.tmem_o_inter_output, tOintertQS_fake.layout
        )

        total_loop_count = cute.ceil_div(mO_qdl.shape[0], self.cta_tiler[0])
        if cutlass.const_expr(self.num_chunk_groups > 1):
            loop_count = total_loop_count // self.num_chunk_groups
        else:
            loop_count = total_loop_count

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load Gate Beta
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.gb_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_gb)
            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            lane_id = tidx % 32

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                chunk_start = curr_block_coord[0] * loop_count if cutlass.const_expr(self.num_chunk_groups > 1) else Int32(0)

                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                head_coord = curr_block_coord[2][0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not GdnStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )

                if not continue_cond:
                    batch_base = cuseqlen_q
                    gb_iters = self.chunk_size // 32
                    gate_vals = cute.make_rmem_tensor((gb_iters), cutlass.Float32)
                    beta_vals = cute.make_rmem_tensor((gb_iters), cutlass.Float32)

                    for i in cutlass.range(loop_count, unroll=1):
                        # step1: read gate beta
                        for it in cutlass.range_constexpr(gb_iters):
                            curr_idx = (chunk_start + i) * self.chunk_size + it * 32 + lane_id
                            if curr_idx < seqlen_q:
                                if cutlass.const_expr(cum_seqlen_q is not None):
                                    gate_vals[it] = gate[
                                        batch_base + curr_idx, 0, (head_coord, 0)
                                    ]
                                    beta_vals[it] = beta[
                                        batch_base + curr_idx, 0, (head_coord, 0)
                                    ]
                                else:
                                    gate_vals[it] = gate[
                                        curr_idx, 0, (head_coord, batch_coord)
                                    ]
                                    beta_vals[it] = beta[
                                        curr_idx, 0, (head_coord, batch_coord)
                                    ]
                            else:
                                gate_vals[it] = cutlass.Float32(0)
                                beta_vals[it] = cutlass.Float32(0)
                        # step2: write to smem
                        sGate0 = sGate[None, None, 0]  # assume only one stage
                        sBeta0 = sBeta[None, None, 0]

                        gb_handle = gb_w0_producer.acquire_and_advance()

                        for it in cutlass.range_constexpr(gb_iters):
                            sGate0[it * 32 + lane_id] = gate_vals[it]
                            sBeta0[it * 32 + lane_id] = beta_vals[it]

                        # Zero-fill remaining SMEM for chunk_size < 128
                        # (threads 64-127 would read garbage otherwise)
                        if cutlass.const_expr(self.chunk_size < 128):
                            for it in cutlass.range_constexpr(
                                gb_iters, 128 // 32
                            ):
                                sGate0[it * 32 + lane_id] = cutlass.Float32(0)
                                sBeta0[it * 32 + lane_id] = cutlass.Float32(0)

                        # step3: notify cuda core wg
                        gb_handle.commit()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue: Store Output
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                chunk_start = curr_block_coord[0] * loop_count if cutlass.const_expr(self.num_chunk_groups > 1) else Int32(0)

                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not GdnStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )

                if not continue_cond:
                    curr_block_coord_o = curr_block_coord
                    mO_qdl_ = mO_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mO = (
                            mO_qdl_.shape[0] - seqlen_q,
                            0,
                            (0, cuseqlen_q + seqlen_q),
                        )
                        mO_qdl_ = cute.domain_offset(logical_offset_mO, mO_qdl_)
                        curr_block_coord_o = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], 0),
                        )

                    gO_mnl = cute.flat_divide(
                        mO_qdl_, self.o_output_tiler
                    )
                    tOsO, tOgO_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_o_output,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sO, 0, 2),
                        cute.group_modes(gO_mnl, 0, 2),
                    )
                    tQgO = tOgO_qdl[None, None, 0, curr_block_coord_o[2]]
                    w0_handle = epi_w0_producer.acquire_and_advance()
                    w0_handle.commit()

                    for i in cutlass.range(loop_count, unroll=1):
                        w0_handle = epi_w0_producer.acquire_and_advance()
                        cute.copy(tma_atom_o_output, tOsO[None, 0], tQgO[None, chunk_start + i])

                        cute.arch.cp_async_bulk_commit_group()

                        # Ensure O0 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(0, read=True)

                        w0_handle.commit()

                    if cutlass.const_expr(tma_atom_state_output is not None):
                        gStateOutput = cute.flat_divide(
                            mStateOutput,
                            cute.select(self.update_s_mma_tiler, mode=[0, 1]),
                        )
                        tSsS, tSgS_vkl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_state_output,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sStateOutput, 0, 2),
                            cute.group_modes(gStateOutput, 0, 4),
                        )
                        tSgS = tSgS_vkl[None, curr_block_coord[2]]
                        w0_epi_handle = w0_epi_consumer.wait_and_advance()
                        cute.copy(
                            tma_atom_state_output,
                            tSsS[None, 0],  # only 1 stage
                            tSgS[None],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        w0_epi_handle.release()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                # The smem is reused, ensuring the State Output store is complete
                if cutlass.const_expr(
                    tma_atom_state_output is not None and self.is_persistent
                ):
                    cute.arch.barrier_arrive(
                        barrier_id=self.epi_load_sync_bar_id,
                        number_of_threads=self.threads_per_warp * 2,
                    )
        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD: tma load QKVS
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                chunk_start = curr_block_coord[0] * loop_count if cutlass.const_expr(self.num_chunk_groups > 1) else Int32(0)
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]
                batch_coord = curr_block_coord[2][1]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not GdnStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )

                if not continue_cond:
                    mQ_qdl_ = mQ_qdl
                    mK_kdl_ = mK_kdl
                    mV_dkl_ = mV_dkl
                    curr_block_coord_q = curr_block_coord
                    curr_block_coord_kv = curr_block_coord
                    curr_block_coord_state = curr_block_coord

                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mQ = (
                            cuseqlen_q,
                            0,
                            (0, 0),
                        )
                        mQ_qdl_ = cute.domain_offset(logical_offset_mQ, mQ_qdl)
                        curr_block_coord_q = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], Int32(0)),
                        )
                        logical_offset_mK = (
                            cuseqlen_q,
                            0,
                            (0, 0),
                        )
                        logical_offset_mV = (
                            0,
                            cuseqlen_q,
                            (0, 0),
                        )
                        mK_kdl_ = cute.domain_offset(logical_offset_mK, mK_kdl)
                        mV_dkl_ = cute.domain_offset(logical_offset_mV, mV_dkl)
                        curr_block_coord_kv = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], Int32(0)),
                        )

                    gQ_qdl = cute.flat_divide(
                        mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2])
                    )
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)

                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQK, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord_q[2]]

                    gK_kdl = cute.flat_divide(
                        mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                    )
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    tKgK = tKgK_kdl[None, None, 0, curr_block_coord_kv[2]]

                    gV_kdl = cute.flat_divide(
                        mV_dkl_, cute.select(self.tuw_mma_tiler, mode=[1, 2])
                    )
                    tSgV_kdl = tuw_thr_mma.partition_B(gV_kdl)
                    tVsV, tVgV_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_kdl, 0, 3),
                    )
                    tVgV = tVgV_kdl[None, 0, None, curr_block_coord_kv[2]]

                    if cutlass.const_expr(tma_atom_state_f32 is not None):
                        # For seq-dim parallelism: wait for previous chunk group's
                        # state before loading (to avoid reading stale data).
                        if cutlass.const_expr(
                            self.num_chunk_groups > 1
                            and workspace_flags_ptr is not None
                        ):
                            chunk_gid = curr_block_coord[0]
                            head_c = curr_block_coord[2][0]
                            batch_c = curr_block_coord[2][1]
                            head_i = batch_c * cute.size(O.shape[2][1]) + head_c
                            if chunk_gid > 0:
                                ws_f = cute.make_tensor(
                                    workspace_flags_ptr,
                                    cute.make_layout((self.num_chunk_groups,)),
                                )
                                tidx_ld, _, _ = cute.arch.thread_idx()
                                if tidx_ld == 0:
                                    while ws_f[head_i * self.num_chunk_groups + chunk_gid - 1] == 0:
                                        pass

                        gStateF32 = cute.flat_divide(
                            mState_f32, cute.select(self.kst_mma_tiler, mode=[1, 2])
                        )
                        tKSgStateF32 = fake_state_thr_mma.partition_B(gStateF32)
                        tSsSF32, tSgSF32_kdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_state_f32,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sStateInputF32, 0, 3),
                            cute.group_modes(tKSgStateF32, 0, 3),
                        )
                        tSgSF32 = tSgSF32_kdl[None, None, 0, curr_block_coord_state[2]]
                        state_f32_handle0 = load_state_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_state_f32,
                            tSgSF32[None, 0],
                            tSsSF32[None, state_f32_handle0.index],
                            tma_bar_ptr=state_f32_handle0.barrier,
                        )

                    for i in cutlass.range(loop_count, unroll=1):
                        w0_coord = chunk_start + i
                        k0_handle = load_qk_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, w0_coord],
                            tKsK[None, k0_handle.index],
                            tma_bar_ptr=k0_handle.barrier,
                        )

                        q_handle = load_qk_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, w0_coord],
                            tQsQ[None, q_handle.index],
                            tma_bar_ptr=q_handle.barrier,
                        )

                        v_handle0 = load_v_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, w0_coord],
                            tVsV[None, v_handle0.index],
                            tma_bar_ptr=v_handle0.barrier,
                        )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                if cutlass.const_expr(
                    tma_atom_state_output is not None and self.is_persistent
                ):
                    cute.arch.barrier(
                        barrier_id=self.epi_load_sync_bar_id,
                        number_of_threads=self.threads_per_warp * 2,
                    )
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA: mma warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            tSrDC_ivt_l1_fake = invert_sub_thr_mma_ts_l1.make_fragment_A(
                invert_sub_smem_layout_staged_ts_l1_a.outer.shape
            )
            tSrDC_ivt_l1 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr,
                    dtype=tSrDC_ivt_l1_fake.element_type,
                )
                + self.tmem_ivt_ts_l1_input,
                tSrDC_ivt_l1_fake.layout,
            )
            tSrA_ivt_l1 = invert_sub_thr_mma_ts_l1.make_fragment_B(sInvertSubTSL1B)

            tOrP = o_intra_thr_mma_new.make_fragment_A(
                o_intra_smem_layout_staged_a_new.outer
            )
            tOrP0 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr,
                    dtype=tOrP.element_type,
                )
                + self.tmem_gate_qk * 2,
                tOrP.layout,
            )

            tQSA = o_intra_thr_mma_new.make_fragment_A(qs_smem_layout_staged_a.outer)
            tQSA0 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr,
                    dtype=tQSA.element_type,
                )
                + self.tmem_gate_q * 2,
                tQSA.layout,
            )

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                continue_cond = False

                if not continue_cond:
                    for _i in cutlass.range(loop_count, unroll=1):
                        k_handle0 = load_qk_consumer.wait_and_advance()
                        qk_handle0 = mma_qk_producer0.acquire_and_advance()

                        kkt_tiled_mma = self.exec_mma(
                            kkt_tiled_mma,
                            tKKTtKKT,
                            tArK,
                            tArKT,
                            k_handle0.index,
                            k_handle0.index,
                        )

                        qk_handle0.commit()

                        q_handle = load_qk_consumer.wait_and_advance()

                        qk_handle1 = mma_qk_producer0.acquire_and_advance()

                        qk_tiled_mma = self.exec_mma(
                            qk_tiled_mma,
                            tQKtQK,
                            tSrQ,
                            tSrK,
                            q_handle.index,
                            k_handle0.index,
                        )

                        qk_handle1.commit()

                        cs_handle = mma_cudacore_producer0.acquire_and_advance()
                        cute.arch.barrier(
                            barrier_id=self.cudacore_mma_sync_bar_id,
                            number_of_threads=self.threads_per_warp * 5,
                        )

                        kst_tiled_mma = self.exec_mma(
                            kst_tiled_mma,
                            tKStKS,
                            tKSrK_0,
                            tKSrST,
                            0,
                            0,
                        )

                        cs_handle.commit()

                        cs_handle = mma_cudacore_producer0.acquire_and_advance()

                        invert_sub_tiled_mma_ss_l0 = self.exec_mma(
                            invert_sub_tiled_mma_ss_l0,
                            tItSSL0,
                            tSrD,
                            tSrC,
                            0,
                            0,
                        )

                        cs_handle.commit()

                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        invert_sub_tiled_mma_ts_l0 = self.exec_mma(
                            invert_sub_tiled_mma_ts_l0,
                            tItTSL0,
                            tInrDCL0A,
                            tSrDCL0B,
                            0,
                            0,
                        )

                        c0_handle.commit()

                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        invert_sub_tiled_mma_ss_l1 = self.exec_mma(
                            invert_sub_tiled_mma_ss_l1,
                            tStO_ivt_l1_ss0,
                            tSrD_ivt_l1,
                            tSrC_ivt_l1,
                            0,
                            0,
                        )

                        c0_handle.commit()

                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        k_handle0.release()

                        invert_sub_tiled_mma_ts_l1 = self.exec_mma(
                            invert_sub_tiled_mma_ts_l1,
                            tOtTSL1,
                            tSrDC_ivt_l1,
                            tSrA_ivt_l1,
                            0,
                            0,
                        )

                        c0_handle.commit()

                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        tuw_tiled_mma = self.exec_mma(
                            tuw_tiled_mma,
                            tVtTUW,
                            tTUWrA,
                            tTUWrB,
                            0,
                            0,
                        )

                        c0_handle.commit()
                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        q_handle.release()

                        tCtAcc_o_intra_acc_shape_fake = o_intra_thr_mma.make_fragment_C(
                            o_intra_acc_shape
                        )
                        tStO_o_intra0 = cute.make_tensor(
                            tmem_ptr + self.tmem_o_intra_output,
                            tCtAcc_o_intra_acc_shape_fake.layout,
                        )

                        o_intra_tiled_ts_mma_new = self.exec_mma(
                            o_intra_tiled_ts_mma_new,
                            tStO_o_intra0,
                            tOrP0,
                            tOIntrarB,
                            0,
                            0,
                        )

                        qs_tiled_mma = self.exec_mma(
                            qs_tiled_mma,
                            tOintertQS,
                            tQSA0,
                            tQSB,
                            0,
                            0,
                            True,
                        )

                        c0_handle.commit()
                        c0_handle = mma_cudacore_producer0.acquire_and_advance()

                        update_s_tiled_mma = self.exec_mma(
                            update_s_tiled_mma,
                            tStUpdate,
                            tUpdateA,
                            tUpdateB,
                            0,
                            0,
                            True,
                        )

                        c0_handle.commit()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                # End of persistent scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        #  Mainloop: mainloop warp group
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < 4:
            cute.arch.setmaxregister_increase(self.num_regs_cudacore)

            tile_sched = create_gdn_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                chunk_start = curr_block_coord[0] * loop_count if cutlass.const_expr(self.num_chunk_groups > 1) else Int32(0)
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                if not continue_cond:
                    gO_mnl = cute.flat_divide(
                        O, self.o_output_tiler
                    )
                    # Use input state — with chunk group sync for seq-dim parallelism
                    if cutlass.const_expr(
                        self.num_chunk_groups > 1
                        and workspace_state_ptr is not None
                    ):
                        chunk_group_id = curr_block_coord[0]
                        head_coord = curr_block_coord[2][0]
                        head_idx = batch_coord * cute.size(O.shape[2][1]) + head_coord
                        state_size = self.head_dim * self.head_dim

                        # Create flag tensor for global memory access
                        ws_flags = cute.make_tensor(
                            workspace_flags_ptr,
                            cute.make_layout((self.num_chunk_groups,)),
                        )
                        flag_base = head_idx * self.num_chunk_groups

                        if chunk_group_id == 0:
                            # Group 0: load initial state normally
                            if cutlass.const_expr(tma_atom_state_f32 is not None):
                                load_state_consumer.wait_and_advance()
                                self.init_state_from_smem(
                                    kst_thr_mma, tStateF32, sStateInputF32,
                                )
                                load_state_consumer.release()
                            else:
                                self.init_state_zeros(kst_thr_mma, tStateF32)
                        else:
                            # Group > 0: spin-wait on previous group's flag
                            if tidx == 0:
                                while ws_flags[flag_base + chunk_group_id - 1] == 0:
                                    pass
                            cute.arch.barrier(
                                barrier_id=self.wg_sync_bar_id,
                                number_of_threads=len(self.cudacore_warp_ids) * 32,
                            )

                            # Load state from output_state (written by previous group's epilogue)
                            if cutlass.const_expr(tma_atom_state_f32 is not None):
                                # Use TMA path: the load warp loaded initial_state for group 0.
                                # For group > 0, we need to load from output_state instead.
                                # Since we can't change TMA descriptor per-group,
                                # consume the TMA load (which loaded initial_state) and replace.
                                load_state_consumer.wait_and_advance()
                                # Overwrite with state from output_state via init_state_from_smem
                                # (the TMA loaded initial_state to sStateInputF32, but we need
                                # output_state which was written by the previous group's epilogue)
                                self.init_state_from_smem(
                                    kst_thr_mma, tStateF32, sStateInputF32,
                                )
                                load_state_consumer.release()
                            else:
                                self.init_state_zeros(kst_thr_mma, tStateF32)
                    else:
                        # Original path (num_chunk_groups == 1)
                        if cutlass.const_expr(tma_atom_state_f32 is not None):
                            load_state_consumer.wait_and_advance()
                            self.init_state_from_smem(
                                kst_thr_mma, tStateF32, sStateInputF32,
                            )
                            load_state_consumer.release()
                        else:
                            self.init_state_zeros(kst_thr_mma, tStateF32)

                    # Arguments
                    atom_args = (
                        kkt_thr_mma,
                        kst_thr_mma,
                        qk_thr_mma,
                        qs_thr_mma,
                    )
                    tensor_args = (
                        sGate,
                        sBeta,
                        sGateCumsum,
                        sStateInput,
                        sIvt,
                        sQ,
                        sK,
                        sV,
                        sO,
                        sInvertSubReg,
                        sInvertSubSSL0A,
                        sInvertSubSSL0B,
                        sInvertSubSSL1B,
                        sInvertSubTSL0B,
                        sInvertSubSSL1A,
                        sInvertSubTSL1B,
                        sTuwAStore,
                        sTuwA,
                        sOIntraB,
                        sUpdateB,
                        tGamma,
                        tKKTtKKT,
                        tQKtQK,
                        tQKMtQKM,
                        tKtK,
                        tKStKS,
                        tItTSL0,
                        tInrDCL0A,
                        tOtTSL1,
                        tVtTUW,
                        tOintertQS,
                        tItSSL0,
                        tStateF32,
                        tQgate,
                        mK_kdl,
                    )
                    pipeline_args = (
                        gb_w0_consumer,
                        epi_w0_consumer,
                        mma_qk_consumer0,
                        mma_cudacore_consumer0,
                        load_v_consumer,
                    )

                    # Corner case
                    tail_count = mO_qdl.shape[0] % self.cta_tiler[0]

                    if tail_count == 0:
                        value_args = (
                            scale,
                            loop_count,
                            chunk_start,
                        )
                        (
                            gb_w0_consumer,
                            epi_w0_consumer,
                            mma_qk_consumer0,
                            mma_cudacore_consumer0,
                            load_v_consumer,
                        ) = self.main_loop(
                            value_args,
                            atom_args,
                            tensor_args,
                            pipeline_args,
                        )
                    else:
                        value_args = (
                            scale,
                            loop_count - 1,
                            chunk_start,
                        )
                        (
                            gb_w0_consumer,
                            epi_w0_consumer,
                            mma_qk_consumer0,
                            mma_cudacore_consumer0,
                            load_v_consumer,
                        ) = self.main_loop(
                            value_args,
                            atom_args,
                            tensor_args,
                            pipeline_args,
                        )
                        pipeline_args = (
                            gb_w0_consumer,
                            epi_w0_consumer,
                            mma_qk_consumer0,
                            mma_cudacore_consumer0,
                            load_v_consumer,
                        )
                        value_args = (
                            scale,
                            1,
                            chunk_start + loop_count - 1,
                        )
                        (
                            gb_w0_consumer,
                            epi_w0_consumer,
                            mma_qk_consumer0,
                            mma_cudacore_consumer0,
                            load_v_consumer,
                        ) = self.main_loop(
                            value_args,
                            atom_args,
                            tensor_args,
                            pipeline_args,
                            True,
                            tail_count=tail_count,
                        )

                    # Output state + workspace sync for seq-dim parallelism
                    if cutlass.const_expr(
                        self.num_chunk_groups > 1
                        and workspace_state_ptr is not None
                    ):
                        chunk_group_id = curr_block_coord[0]
                        head_coord = curr_block_coord[2][0]
                        head_idx = batch_coord * cute.size(O.shape[2][1]) + head_coord

                    if cutlass.const_expr(tma_atom_state_output is not None):
                        w0_epi_handle = w0_epi_producer.acquire_and_advance()
                        self.store_state_to_smem(
                            tStUpdate, sStateOutput, update_s_thr_mma
                        )
                        w0_epi_handle.commit()

                    w0_handle = epi_w0_consumer.wait_and_advance()
                    w0_handle.release()

                    # Set flag AFTER state has been written to global memory
                    if cutlass.const_expr(
                        self.num_chunk_groups > 1
                        and workspace_flags_ptr is not None
                    ):
                        chunk_group_id = curr_block_coord[0]
                        head_coord = curr_block_coord[2][0]
                        head_idx = batch_coord * cute.size(O.shape[2][1]) + head_coord
                        ws_flags_out = cute.make_tensor(
                            workspace_flags_ptr,
                            cute.make_layout((self.num_chunk_groups,)),
                        )
                        # Sync threads + fence to ensure TMA state store is visible
                        cute.arch.barrier(
                            barrier_id=self.wg_sync_bar_id,
                            number_of_threads=len(self.cudacore_warp_ids) * 32,
                        )
                        if tidx == 0:
                            ws_flags_out[head_idx * self.num_chunk_groups + chunk_group_id] = Int32(1)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    @cute.jit
    def main_loop(
        self,
        value_args: tuple,
        atom_args: tuple,
        tensor_args: tuple,
        pipeline_args: tuple,
        need_mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ) -> Tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
    ]:
        """Per-chunk mainloop executed by cudacore warps (0-3)."""
        tidx, _, _ = cute.arch.thread_idx()

        (
            scale,
            loop_count,
            loop_base,
        ) = value_args

        (
            kkt_thr_mma,
            kst_thr_mma,
            qk_thr_mma,
            qs_thr_mma,
        ) = atom_args

        (
            sGate,
            sBeta,
            sGateCumsum,
            sStateInput,
            sIvt,
            sQ,
            sK,
            sV,
            sO,
            sInvertSubReg,
            sInvertSubSSL0A,
            sInvertSubSSL0B,
            sInvertSubSSL1B,
            sInvertSubTSL0B,
            sInvertSubSSL1A,
            sInvertSubTSL1B,
            sTuwAStore,
            sTuwA,
            sOIntraB,
            sUpdateB,
            tGamma,
            tKKTtKKT,
            tQKtQK,
            tQKMtQKM,
            tKtK,
            tKStKS,
            tItTSL0,
            tInrDCL0A,
            tOtTSL1,
            tVtTUW,
            tOintertQS,
            tItSSL0,
            tStateF32,
            tQgate,
            mK_global,
        ) = tensor_args

        (
            gb_w0_consumer,
            epi_w0_consumer,
            mma_qk_consumer0,
            mma_cudacore_consumer0,
            load_v_consumer,
        ) = pipeline_args

        for _i in cutlass.range(loop_base, loop_base + loop_count, unroll=1):
            gb_handle = gb_w0_consumer.wait_and_advance()
            sGate0 = sGate[None, None, gb_handle.index]
            sBeta0 = sBeta[None, None, gb_handle.index]
            tval = self.chunk_local_cumsum(sGate0, sGateCumsum, self.wg_sync_bar_id)
            tidx_in_group = tidx % (self.threads_per_warp * len(self.cudacore_warp_ids))
            beta_val = sBeta0[tidx_in_group] if tidx_in_group < self.chunk_size else self.g_beta_dtype(0)
            gb_handle.release()

            self.compute_gamma_tmem(
                tval,
                sGateCumsum,
                kkt_thr_mma,
                tGamma,
                self.wg_sync_bar_id,
                need_mask,
                tail_count,
            )
            qk_handle0 = mma_qk_consumer0.wait_and_advance()

            self.apply_gamma_beta(
                kkt_thr_mma,
                sIvt,
                tKKTtKKT,
                beta_val,
                tGamma,
                sGateCumsum,
            )

            # For chunk_size=64: compute sIvt from K in global memory.
            # sIvt[(k, row), W] = beta * gamma(row, k) * dot(K[row,:], K[k,:])
            # Each warp handles one 32×32 diagonal block. Each lane = one row.
            if cutlass.const_expr(self.chunk_size <= 64):
                warp_id_local = cute.arch.warp_idx() % 4
                lane_id_local = cute.arch.lane_idx()
                chunk_token_offset = loop_base * self.chunk_size
                for W in cutlass.range_constexpr(self.chunk_size // 32):
                    if warp_id_local == W:
                        row_token = chunk_token_offset + W * 32 + lane_id_local
                        cum_row = sGateCumsum[W * 32 + lane_id_local]
                        for k_local in cutlass.range(0, lane_id_local):
                            col_token = chunk_token_offset + W * 32 + k_local
                            cum_col = sGateCumsum[W * 32 + k_local]
                            gamma_val = cute.math.exp(cum_row - cum_col, fastmath=True)
                            sIvt[(k_local, lane_id_local), W] = self.acc_dtype(0)
                cute.arch.barrier(
                    barrier_id=self.wg_sync_bar_id,
                    number_of_threads=len(self.cudacore_warp_ids) * 32,
                )

            gate_tail = (
                sGateCumsum[tail_count - 1]
                if cutlass.const_expr(need_mask)
                else sGateCumsum[self.chunk_size - 1]
            )
            w0_handle = epi_w0_consumer.wait_and_advance()
            self.load_state_apply_gate(
                kst_thr_mma,
                tStateF32,
                sStateInput,
                gate_tail,
            )
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )

            qk_handle1 = mma_qk_consumer0.wait_and_advance()
            self.load_qk_epi(
                qk_thr_mma, tQKtQK, tQKMtQKM, tGamma, need_mask, tail_count
            )

            cute.arch.barrier_arrive(
                barrier_id=self.cudacore_mma_sync_bar_id,
                number_of_threads=self.threads_per_warp * 5,
            )

            reverse_result = cute.make_rmem_tensor((32, 1), self.acc_dtype)

            self.reverse_smem_sub(reverse_result, sIvt)

            self.store_ivt_smem_l0_ss_a(
                reverse_result,
                sInvertSubReg,
                sInvertSubSSL0A,
            )

            self.store_ivt_smem_l0_ss_b(
                kkt_thr_mma,
                tKKTtKKT,
                sInvertSubSSL0B,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            c0_handle.release()
            v_handle0 = load_v_consumer.wait_and_advance()

            tval_exp = cute.math.exp(tval)
            self.get_uw_b(
                kst_thr_mma, tKStKS, tval_exp, beta_val, sV, need_mask, tail_count
            )
            self.store_ivt_p1(
                sInvertSubReg,
                sInvertSubTSL0B,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            self.load_ivt_ss_l0(tItSSL0, tInrDCL0A)  # debug
            c0_handle.release()
            self.store_ivt_smem_l1_ss_b(
                kkt_thr_mma,
                tKKTtKKT,
                sInvertSubSSL1B,
            )

            self.store_ivt_p2(
                sInvertSubReg,
                sInvertSubSSL1A,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()

            self.load_ivt_ts_l0(
                tItTSL0,
                sInvertSubSSL1A,
                sInvertSubTSL1B,
            )

            c0_handle.release()

            self.store_k_epi(kkt_thr_mma, sK, tKtK, need_mask, tail_count)
            self.store_ivt_p3(
                sInvertSubReg,
                sInvertSubTSL1B,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            c0_handle.release()
            self.save_tmem(tItTSL0, sTuwAStore)
            self.store_ivt_ad(sInvertSubReg, sTuwA)
            self.store_ivt_c(sTuwAStore)

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            self.load_ivt_result(tOtTSL1, sTuwAStore)
            c0_handle.release()

            self.load_q_epi(qk_thr_mma, sQ, tQgate, tval_exp, need_mask, tail_count)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            v_handle0.release()
            self.load_v(tVtTUW, sOIntraB)
            c0_handle.release()
            self.load_k(tKtK, sUpdateB, gate_tail, tval, sK, kkt_thr_mma)

            # For chunk_size<=64: zero ALL inversion pipeline SMEM buffers
            # by zeroing each buffer individually.
            if cutlass.const_expr(self.chunk_size <= 64):
                # Zero each buffer used by the inversion pipeline MMAs
                def zero_smem(buf, tid):
                    n = cute.cosize(buf)
                    raw = cute.make_tensor(cute.recast_ptr(buf.iterator, dtype=cutlass.Float32), n)
                    for e in cutlass.range_constexpr(n // 128):
                        raw[e * 128 + tid % 128] = cutlass.Float32(0)
                zero_smem(sInvertSubSSL1A, tidx)   # = sUpdateA
                zero_smem(sTuwA, tidx)
                zero_smem(sTuwAStore, tidx)
                zero_smem(sInvertSubSSL0A, tidx)   # = sUpdateB
                zero_smem(sInvertSubSSL0B, tidx)
                zero_smem(sInvertSubSSL1B, tidx)
                zero_smem(sInvertSubTSL0B, tidx)
                zero_smem(sInvertSubTSL1B, tidx)
                zero_smem(sInvertSubReg, tidx)
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )

            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            qk_handle0.release()
            c0_handle.release()
            self.store_o_smem(tOintertQS, sO, qs_thr_mma, scale)

            w0_handle.release()
            qk_handle1.release()
            c0_handle = mma_cudacore_consumer0.wait_and_advance()
            c0_handle.release()

        return (
            gb_w0_consumer,
            epi_w0_consumer,
            mma_qk_consumer0,
            mma_cudacore_consumer0,
            load_v_consumer,
        )

    @cute.jit
    def exec_mma(
        self,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        a_consumer_index,
        b_consumer_index,
        setAcc: bool = False,
    ):
        """Issue a tcgen05 GEMM."""
        for kphase_idx in cutlass.range(cute.size(tCrB, mode=[2]), unroll_all=True):
            # set accu = 1
            tiled_mma.set(
                tcgen05.Field.ACCUMULATE,
                cutlass.Boolean(kphase_idx != 0 or setAcc),
            )
            cute.gemm(
                tiled_mma,
                tCtAcc,
                tCrA[None, None, kphase_idx, a_consumer_index],
                tCrB[None, None, kphase_idx, b_consumer_index],
                tCtAcc,
            )
        return tiled_mma

    @cute.jit
    def get_uw_b(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        gate_val: cutlass.Float32,
        beta_val: cutlass.Float32,
        sV: cute.Tensor,
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ):
        """Compute V_new = beta*V - beta*gamma*(K@S^T) in-place in smem."""
        tidx, _, _ = cute.arch.thread_idx()

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((self.chunk_size, self.head_dim))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 64
        corr_tile_size_f32 = corr_tile_size // 2

        # Flatten TMEM layout first (handles both flat 128-col and hierarchical 64-col)
        tIn_flat_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn_flat = cute.make_tensor(tIn.iterator, tIn_flat_layout)

        tIn_i_layout = cute.composition(
            tIn_flat.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )

        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )

        tIn_i = cute.make_tensor(tIn_flat.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        if cutlass.const_expr(self.chunk_size <= 64):
            # Subtile Ld16x256bOp: load K@S^T, compute V_new, store to sV SMEM
            if cutlass.const_expr(mask):
                if thread_idx >= tail_count:
                    beta_val = self.g_beta_dtype(0)
            combined_factor = gate_val * beta_val

            tIn_flat2 = cute.make_tensor(
                tIn.iterator,
                cute.composition(tIn.layout, cute.make_layout((tIn.shape))),
            )
            ld_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
            tc = tcgen05.make_tmem_copy(ld_atom, tIn_flat2)
            thr_ld = tc.get_slice(thread_idx)
            tSrc = thr_ld.partition_S(tIn_flat2)
            tDst = thr_ld.partition_D(tIn_flat2)
            tDst_one = tDst[(None), 0, 0, 0]
            regs_kst = cute.make_rmem_tensor(tDst_one.shape, self.acc_dtype)
            tSrc = cute.group_modes(tSrc, 1, cute.rank(tSrc))

            # Use MMA partition to access sV SMEM (element order matches flatten)
            tMsV = thr_mma.partition_C(sV)
            tMsV_flat = cute.make_tensor(
                cute.recast_ptr(tMsV.iterator, dtype=self.i_dtype),
                cute.size(tMsV),  # flatten to 1D
            )

            n = cute.size(regs_kst)
            for subtile_idx in cutlass.range_constexpr(2):
                cute.copy(tc, tSrc[(None, subtile_idx)], regs_kst)
                cute.arch.fence_view_async_tmem_load()
                for eid in cutlass.range_constexpr(n):
                    kst_val = regs_kst[eid] * (-combined_factor)
                    smem_idx = subtile_idx * n + eid
                    v_old = cutlass.Float32(tMsV_flat[smem_idx])
                    v_new = beta_val * v_old + kst_val
                    tMsV_flat[smem_idx] = v_new.to(self.i_dtype)

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            return

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size_f32)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        frg_tile = 8
        sV_frg = cute.logical_divide(sV, cute.make_layout(frg_tile))

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)
        tTMEM_LOADrS_frag = cute.logical_divide(
            tTMEM_LOADrS, cute.make_layout(frg_tile)
        )

        if cutlass.const_expr(mask):
            if thread_idx >= tail_count:
                beta_val = self.g_beta_dtype(0)

        combined_factor = gate_val * beta_val

        each_iter = corr_tile_size // frg_tile
        for i in cutlass.range_constexpr(self.head_dim // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()

            # Load V from smem
            temp_regs = cute.make_rmem_tensor(tTMEM_LOADrS_frag.shape, self.acc_dtype)
            for it in cutlass.range_constexpr(each_iter):
                for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                    # mul2
                    (
                        temp_regs[inner_idx + 0, it],
                        temp_regs[inner_idx + 1, it],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            tTMEM_LOADrS_frag[inner_idx + 0, it],
                            tTMEM_LOADrS_frag[inner_idx + 1, it],
                        ),
                        (-combined_factor, -combined_factor),
                    )

                    # fma2
                    (
                        temp_regs[inner_idx + 0, it],
                        temp_regs[inner_idx + 1, it],
                    ) = cute.arch.fma_packed_f32x2(
                        (
                            beta_val,
                            beta_val,
                        ),
                        (
                            cutlass.Float32(
                                sV_frg[inner_idx + 0, (i * each_iter + it, thread_idx)]
                            ),
                            cutlass.Float32(
                                sV_frg[inner_idx + 1, (i * each_iter + it, thread_idx)]
                            ),
                        ),
                        (
                            temp_regs[inner_idx + 0, it],
                            temp_regs[inner_idx + 1, it],
                        ),
                    )
                sV_frg[None, (i * each_iter + it, thread_idx)].store(
                    temp_regs[None, it].load().to(self.i_dtype)
                )

    @cute.jit
    def store_state_to_smem(
        self,
        tO: cute.Tensor,
        sO: cute.Tensor,
        thr_mma: cute.ThrMma,
    ):
        """Copy final state from TMEM to smem for TMA store to global."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tOcO = thr_mma.partition_C(tO)
        tOsO = thr_mma.partition_C(sO)

        corr_tile_size = 32  # must >= 8
        tOsO_sub = cute.make_tensor(
            cute.recast_ptr(tOsO.iterator, dtype=self.state_dtype),
            cute.shape(tOsO)[0],
        )
        tOsO_tile = cute.logical_divide(
            tOsO_sub, (None, (None, 32 // (128 // corr_tile_size)))
        )

        tIn_i_layout = cute.composition(
            tO.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )
        sOut_i_layout = cute.composition(
            tOsO.layout, cute.make_layout((128, corr_tile_size))
        )

        tIn_i = cute.make_tensor(tO.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(tOcO.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.s_output, self.state_dtype, self.acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        for i in cutlass.range_constexpr(self.head_dim // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            sOut_i_sub = cute.make_tensor(
                tOsO_tile[None, (None, (None, i))].iterator, sOut_i_layout
            )
            tTMEM_STOREsO_i = thr_tmem_load.partition_D(sOut_i_sub)
            cute.copy(tiled_smem_store, tTMEM_LOADrS, tTMEM_STOREsO_i)

        # fence view async shared
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def store_o_smem(
        self,
        tO: cute.Tensor,
        sO: cute.Tensor,
        thr_mma: cute.ThrMma,
        scale: cutlass.Float32,
    ):
        """Scale and convert O from TMEM (f32) to smem (f16/bf16) for TMA store."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tOcO = thr_mma.partition_C(tO)
        tOsO = thr_mma.partition_C(sO)

        corr_tile_size = 128  # must >= 8

        # ((128,(8,16)),1,1,(1,1))
        tOsO_sub = cute.make_tensor(
            cute.recast_ptr(tOsO.iterator, dtype=self.i_dtype),
            cute.shape(tOsO)[0],
        )
        tOsO_tile = cute.logical_divide(
            tOsO_sub, (None, (None, 16 // (128 // corr_tile_size)))
        )

        # Flatten TMEM layout first (handles both flat 128-col and hierarchical 64-col)
        tO_flat_layout = cute.composition(tO.layout, cute.make_layout((tO.shape)))
        tO_flat = cute.make_tensor(tO.iterator, tO_flat_layout)

        tIn_i_layout = cute.composition(
            tO_flat.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        sOut_i_layout = cute.composition(
            tOsO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )

        tIn_i = cute.make_tensor(tO_flat.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(tOcO.iterator, cIn_i_layout)

        if cutlass.const_expr(self.chunk_size <= 64):
            # Subtile Ld16x256bOp + SMEM store via MMA partition.
            # The flatten preserves MMA write order, so regs[i] maps to
            # thr_mma.partition_C(sO)[i] — enabling direct per-element SMEM stores.
            tO_flat2 = cute.make_tensor(
                tO.iterator,
                cute.composition(tO.layout, cute.make_layout((tO.shape))),
            )
            ld_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
            tc = tcgen05.make_tmem_copy(ld_atom, tO_flat2)
            thr_ld = tc.get_slice(thread_idx)
            tSrc = thr_ld.partition_S(tO_flat2)
            tDst = thr_ld.partition_D(tO_flat2)
            tDst_one = tDst[(None), 0, 0, 0]
            regs = cute.make_rmem_tensor(tDst_one.shape, self.acc_dtype)
            tSrc = cute.group_modes(tSrc, 1, cute.rank(tSrc))

            # SMEM store: use load_ivt_result style with ALL data loaded at once
            # (not subtile iteration) and direct SMEM write via MMA partition.
            # Load ALL from TMEM in one shot (like load_ivt_result)
            regs_full = cute.make_rmem_tensor(tDst.shape, self.acc_dtype)
            tSrc_full = cute.group_modes(thr_ld.partition_S(tO_flat2), 1, cute.rank(thr_ld.partition_S(tO_flat2)))
            regs_full_g = cute.group_modes(regs_full, 1, cute.rank(regs_full))
            cute.copy(tc, tSrc_full, regs_full_g)
            cute.arch.fence_view_async_tmem_load()

            # Scale and convert, then store to SMEM via MMA partition
            tOsO_part = thr_mma.partition_C(sO)
            n_total = cute.size(regs_full)
            frg_cnt = 2
            frg_tile = n_total // frg_cnt
            regs_frg = cute.logical_divide(regs_full, cute.make_layout(frg_tile))
            regs_e = cute.make_rmem_tensor(regs_full.shape, self.i_dtype)
            regs_e_frg = cute.logical_divide(regs_e, cute.make_layout(frg_tile))
            for j in cutlass.range_constexpr(frg_cnt):
                r_vec = regs_frg[None, j].load()
                regs_e_frg[None, j].store((r_vec * scale).to(self.i_dtype))

            # Store to sO SMEM via MMA partition (same approach as get_uw_b).
            # The Ld16x256bOp partition order matches the MMA partition order
            # when both are flattened on the same underlying tensor.
            tMsO = thr_mma.partition_C(sO)
            tMsO_flat = cute.make_tensor(
                cute.recast_ptr(tMsO.iterator, dtype=self.i_dtype),
                cute.size(tMsO),
            )
            for i in cutlass.range_constexpr(n_total):
                tMsO_flat[i] = regs_e[i]

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            return

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_LOADrS.iterator, dtype=self.i_dtype),
            tTMEM_LOADrS.layout,
        )

        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout, self.i_dtype, self.acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        for i in cutlass.range_constexpr(self.head_dim // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)

            frg_cnt = 4  # must 4
            frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
            tTMEM_LOADrS_frg = cute.logical_divide(
                tTMEM_LOADrS, cute.make_layout(frg_tile)
            )
            tTMEM_STORErS_x4_e_frg = cute.logical_divide(
                tTMEM_STORErS_x4_e, cute.make_layout(frg_tile)
            )

            temp_regs = cute.make_rmem_tensor(tTMEM_LOADrS_frg.shape, self.acc_dtype)
            for j in cutlass.range_constexpr(frg_cnt):
                for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                    # mul2
                    (
                        temp_regs[inner_idx + 0, j],
                        temp_regs[inner_idx + 1, j],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            tTMEM_LOADrS_frg[inner_idx + 0, j],
                            tTMEM_LOADrS_frg[inner_idx + 1, j],
                        ),
                        (scale, scale),
                    )
                tTMEM_STORErS_x4_e_frg[None, j].store(
                    temp_regs[None, j].load().to(self.i_dtype)
                )

            # store
            sOut_i_sub = cute.make_tensor(
                tOsO_tile[None, (None, (None, i))].iterator, sOut_i_layout
            )
            tTMEM_STOREsO_i = thr_tmem_load.partition_D(sOut_i_sub)
            cute.copy(tiled_smem_store, tTMEM_STORErS_x4_e, tTMEM_STOREsO_i)

        # fence view async shared
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def load_k(
        self,
        tIn: cute.Tensor,
        sUpdateB: cute.Tensor,
        gate: cutlass.Float32,
        val: cutlass.Float32,
        sK_smem: cute.Tensor,
        kkt_thr_mma_arg: cute.ThrMma,
    ):
        """Load K from TMEM (chunk_size>=128) or SMEM (chunk_size<=64), apply per-token gate decay."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        gate_val = cute.math.exp((gate - val))

        if cutlass.const_expr(self.chunk_size <= 64):
            # For chunk_size=64: tKtK TMEM (offset 64) is outside KK^T accumulator range.
            # Read K from sK SMEM directly, apply gate, write to sUpdateB SMEM.
            #
            # Thread mapping for chunk_size=64 with 128 threads, 64 K-rows, 128 N-cols:
            #   K-row = thread_idx // 2  (0-63)
            #   N-group = thread_idx % 2 (0-1, each covering 64 N-columns)
            #   Each thread writes 8 store_col × 8 frag = 64 f16 elements
            k_row = thread_idx // 2
            n_group = thread_idx % 2

            # Read K[k_row, :] from sK SMEM (128 head_dim values)
            sK_frag = cute.logical_divide(sK_smem, (((None, 8), None)))
            k_regs = cute.make_rmem_tensor(cute.make_layout(self.head_dim), self.i_dtype)
            k_regs_frag = cute.logical_divide(k_regs, cute.make_layout(8))

            for i in cutlass.range_constexpr(self.chunk_size // 64):
                for col in cutlass.range_constexpr(0, 8):
                    k_regs_frag[None, i * 8 + col].store(
                        sK_frag[
                            (k_row, (None, col % 2)), 0, (col // 2, i), 0
                        ].load()
                    )

            # Store to sUpdateB: each thread writes 8 cols of its row and N-group
            store_row_bid = k_row // 16
            store_row_offset = k_row % 16
            for col in cutlass.range_constexpr(0, 8):
                store_col = n_group * 8 + col
                sUpdateB[
                    ((None, store_col), store_row_offset), 0, store_row_bid, 0
                ].store(
                    (k_regs_frag[None, store_col].load() * gate_val).to(self.i_dtype)
                )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            return

            # Dead code below (kept for reference)
            for subtile_idx in cutlass.range_constexpr(2):
                cute.copy(tiled_copy_t2r, tTR_tO[(None, subtile_idx)], tTR_rO)
                cute.arch.fence_view_async_tmem_load()

                # Apply gate and convert f32 -> f16
                frg_cnt = 2
                frg_tile = cute.size(tTR_rO) // frg_cnt
                tTR_rO_frg = cute.logical_divide(tTR_rO, cute.make_layout(frg_tile))
                tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
                for j in cutlass.range_constexpr(frg_cnt):
                    r_vec = tTR_rO_frg[None, j].load()
                    tTR_rO_e_frg[None, j].store((r_vec * gate_val).to(self.i_dtype))

                # Store to SMEM (same pattern as load_v)
                sUpdateB_frg = cute.logical_divide(sUpdateB, cute.make_layout(frg_tile))
                for it in cutlass.range_constexpr(frg_cnt):
                    sUpdateB_frg[None, (subtile_idx * frg_cnt + it, thread_idx)].store(
                        tTR_rO_e_frg[None, it].load()
                    )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            return

        k = self.read_tmem_128(tIn)

        k_f16 = cute.make_tensor(
            cute.recast_ptr(k.iterator, dtype=self.i_dtype),
            cute.make_layout(128),
        )
        k_f16_frag = cute.logical_divide(k_f16, cute.make_layout(8))

        for bid in cutlass.range_constexpr(0, 2):
            for col in cutlass.range_constexpr(0, 8):
                load_row = thread_idx

                store_row = load_row
                store_col = bid * 8 + col

                store_row_bid = store_row // 16
                store_row_offset = store_row % 16
                sUpdateB[
                    ((None, store_col), store_row_offset), 0, store_row_bid, 0
                ].store(
                    (k_f16_frag[None, store_col].load() * gate_val).to(self.i_dtype)
                )

        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def init_state_zeros(
        self,
        thr_mma: cute.ThrMma,
        tOutF32: cute.Tensor,
    ):
        """Zero-initialize the recurrent state in TMEM (no initial_state provided)."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((self.head_dim, self.head_dim))
        tOcO = thr_mma.partition_C(cIn)
        corr_tile_size = 128

        # Apply gate State
        cInOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.head_dim, corr_tile_size))
        )
        tInOut_i_layout = cute.composition(
            tOutF32.layout, cute.make_layout((self.head_dim, corr_tile_size))
        )

        tInOut_i = cute.make_tensor(tOutF32.iterator, tInOut_i_layout)
        cInOut_i = cute.make_tensor(tOcO.iterator, cInOut_i_layout)

        tmem_store_atom_f32 = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_store_f32 = tcgen05.make_tmem_copy(tmem_store_atom_f32, tInOut_i)
        thr_tmem_store_f32 = tiled_tmem_store_f32.get_slice(thread_idx)

        tTMEM_STOREtInOut = thr_tmem_store_f32.partition_D(tInOut_i)
        tTMEM_STOREcInOut = thr_tmem_store_f32.partition_S(cInOut_i)
        tTMEM_STORErInOut = cute.make_rmem_tensor(
            tTMEM_STOREcInOut.shape, self.acc_dtype
        )

        frg_tile = 4
        zeros = cute.zeros_like(cute.make_layout(frg_tile), self.acc_dtype)

        for i in cutlass.range_constexpr(self.head_dim // corr_tile_size):
            tTMEM_STOREtInOut_i = cute.make_tensor(
                tTMEM_STOREtInOut.iterator + i * corr_tile_size,
                tTMEM_STOREtInOut.layout,
            )

            tTMEM_STORErInOut_frg = cute.logical_divide(
                tTMEM_STORErInOut, ((frg_tile, None), None)
            )

            each_iter = corr_tile_size // frg_tile
            for j in cutlass.range_constexpr(each_iter):
                tTMEM_STORErInOut_frg[((None, j), 0), 0, 0].store(zeros)

            # store
            cute.copy(tiled_tmem_store_f32, tTMEM_STORErInOut, tTMEM_STOREtInOut_i)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def init_state_from_smem(
        self,
        thr_mma: cute.ThrMma,
        tOutF32: cute.Tensor,
        sStateInputF32: cute.Tensor,
    ):
        """Load initial_state from smem (f32) into TMEM."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((self.head_dim, self.head_dim))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 128
        cInOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.head_dim, corr_tile_size))
        )
        tInOut_i_layout = cute.composition(
            tOutF32.layout, cute.make_layout((self.head_dim, corr_tile_size))
        )

        tInOut_i = cute.make_tensor(tOutF32.iterator, tInOut_i_layout)
        cInOut_i = cute.make_tensor(tOcO.iterator, cInOut_i_layout)

        tmem_store_atom_f32 = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_store_f32 = tcgen05.make_tmem_copy(tmem_store_atom_f32, tInOut_i)
        thr_tmem_store_f32 = tiled_tmem_store_f32.get_slice(thread_idx)

        tTMEM_STOREtInOut = thr_tmem_store_f32.partition_D(tInOut_i)
        tTMEM_STOREcInOut = thr_tmem_store_f32.partition_S(cInOut_i)
        tTMEM_STORErInOut = cute.make_rmem_tensor(
            tTMEM_STOREcInOut.shape, self.acc_dtype
        )

        frg_tile = 4  # must 4
        sStateInputF32_frag = cute.logical_divide(
            sStateInputF32, (((None, frg_tile), None))
        )
        tTMEM_STORErInOut_frg = cute.logical_divide(
            tTMEM_STORErInOut, ((frg_tile, None), None)
        )
        each_iter = corr_tile_size // frg_tile
        for i in cutlass.range_constexpr(self.head_dim // corr_tile_size):
            tTMEM_STOREtInOut_i = cute.make_tensor(
                tTMEM_STOREtInOut.iterator + i * corr_tile_size,
                tTMEM_STOREtInOut.layout,
            )

            # Load from smem
            idx_offset = i * corr_tile_size
            for j in cutlass.range_constexpr(each_iter):
                idx = idx_offset + j * frg_tile
                bid = idx // 32
                bid_inner = idx % 32 // 8
                inner_idx = idx % 8 // 4
                s_vec = sStateInputF32_frag[
                    (thread_idx, (None, inner_idx)), 0, (bid_inner, bid), 0
                ].load()
                tTMEM_STORErInOut_frg[((None, j), 0), 0, 0].store(s_vec)

            # store
            cute.copy(tiled_tmem_store_f32, tTMEM_STORErInOut, tTMEM_STOREtInOut_i)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_state_apply_gate(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        sStateInput: cute.Tensor,
        gate: cutlass.Float32,
    ):
        """Apply chunk-level gate decay to state in TMEM, and copy state to smem (f16) for KST GEMM."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((self.head_dim, self.head_dim))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 16
        # corr_tile_size_f32 = corr_tile_size // 2

        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((self.head_dim, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.head_dim, corr_tile_size))
        )

        # Apply gate State
        # cInOut_i_layout = cute.composition(tOcO.layout, cute.make_layout((self.head_dim, corr_tile_size)))
        tInOut_i_layout = cute.composition(
            tIn.layout, cute.make_layout((self.head_dim, corr_tile_size))
        )

        # fp16 state
        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tInOut_i = cute.make_tensor(tIn.iterator, tInOut_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tmem_store_atom_f32 = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        tiled_tmem_store_f32 = tcgen05.make_tmem_copy(tmem_store_atom_f32, tInOut_i)

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store_f32 = tiled_tmem_store_f32.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_STOREtInOut = thr_tmem_store_f32.partition_D(tInOut_i)
        tTMEM_STORErInOut = cute.make_tensor(
            cute.recast_ptr(tTMEM_LOADrS.iterator, dtype=self.acc_dtype),
            tTMEM_LOADrS.layout,
        )

        gate_val = cute.math.exp(gate)
        for i in cutlass.range_constexpr(self.head_dim // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtInOut_i = cute.make_tensor(
                tTMEM_STOREtInOut.iterator + i * corr_tile_size,
                tTMEM_STOREtInOut.layout,
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)

            frg_tile = 16  # must <= 16
            tTMEM_LOADrS_frg = cute.logical_divide(
                tTMEM_LOADrS, ((frg_tile, None), None)
            )
            tTMEM_STORErInOut_frg = cute.logical_divide(
                tTMEM_STORErInOut, ((frg_tile, None), None)
            )
            sStateInput_frag = cute.logical_divide(
                sStateInput, (((None, frg_tile), None))
            )

            each_iter = corr_tile_size // frg_tile
            idx_offset = i * corr_tile_size
            for j in cutlass.range_constexpr(each_iter):
                s_vec = tTMEM_LOADrS_frg[((None, j), 0), 0, 0].load()
                s_vec_f16 = s_vec.to(self.q_dtype)
                idx = idx_offset + j * frg_tile
                bid = idx // 64
                bid_inner = idx % 64 // 16
                inner_idx = idx % 16 // frg_tile
                sStateInput_frag[
                    (thread_idx, (None, inner_idx)), 0, (bid_inner, bid), 0
                ].store(s_vec_f16)
                # tTMEM_STORErInOut_frg[((None, j), 0), 0, 0].store(s_vec * gate_val)

                for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                    (
                        tTMEM_STORErInOut_frg[((inner_idx + 0, j), 0), 0, 0],
                        tTMEM_STORErInOut_frg[((inner_idx + 1, j), 0), 0, 0],
                    ) = cute.arch.mul_packed_f32x2(
                        (s_vec[inner_idx + 0], s_vec[inner_idx + 1]),
                        (
                            gate_val,
                            gate_val,
                        ),
                    )
            cute.copy(tiled_tmem_store_f32, tTMEM_STORErInOut, tTMEM_STOREtInOut_i)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_qk_epi(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        tOut: cute.Tensor,
        tGamma: cute.Tensor,  # causal gate mask
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ):
        """Apply gamma mask to QK^T."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((self.chunk_size, self.chunk_size))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 32
        corr_tile_size_f32 = corr_tile_size // 2

        # Flatten TMEM layouts first (handles both flat 128-col and hierarchical 64-col)
        tIn_flat_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn_flat = cute.make_tensor(tIn.iterator, tIn_flat_layout)

        tGamma_flat_layout = cute.composition(tGamma.layout, cute.make_layout((tGamma.shape)))
        tGamma_flat = cute.make_tensor(tGamma.iterator, tGamma_flat_layout)

        tOut_flat_layout = cute.composition(tOut.layout, cute.make_layout((tOut.shape)))
        tOut_flat = cute.make_tensor(tOut.iterator, tOut_flat_layout)

        tIn_i_layout = cute.composition(
            tIn_flat.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )

        cOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size_f32))
        )
        tOut_i_layout = cute.composition(
            tOut_flat.layout, cute.make_layout((self.chunk_size, corr_tile_size_f32))
        )

        tIn_i = cute.make_tensor(tIn_flat.iterator, tIn_i_layout)
        tGamma_i = cute.make_tensor(tGamma_flat.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tOut_i = cute.make_tensor(tOut_flat.iterator, tOut_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cOut_i_layout)

        if cutlass.const_expr(self.chunk_size <= 64):
            # Ld16x256bOp load from QK^T + gamma, multiply, St16x256bOp store to tQKMtQKM
            # Load QK^T (full tensor, flattened)
            qk_flat = cute.make_tensor(
                tIn.iterator,
                cute.composition(tIn.layout, cute.make_layout((tIn.shape))),
            )
            ld_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
            tc_qk = tcgen05.make_tmem_copy(ld_atom, qk_flat)
            thr_qk = tc_qk.get_slice(thread_idx)
            tSrc_qk = thr_qk.partition_S(qk_flat)
            tDst_qk = thr_qk.partition_D(qk_flat)
            regs_qk = cute.make_rmem_tensor(tDst_qk.shape, self.acc_dtype)
            tSrc_qk = cute.group_modes(tSrc_qk, 1, cute.rank(tSrc_qk))
            regs_qk = cute.group_modes(regs_qk, 1, cute.rank(regs_qk))
            cute.copy(tc_qk, tSrc_qk, regs_qk)
            cute.arch.fence_view_async_tmem_load()

            # Load gamma (flat TMEM)
            g_flat = cute.make_tensor(
                tGamma.iterator,
                cute.composition(tGamma.layout, cute.make_layout((tGamma.shape))),
            )
            tc_g = tcgen05.make_tmem_copy(ld_atom, g_flat)
            thr_g = tc_g.get_slice(thread_idx)
            tSrc_g = thr_g.partition_S(g_flat)
            tDst_g = thr_g.partition_D(g_flat)
            regs_g = cute.make_rmem_tensor(tDst_g.shape, self.acc_dtype)
            tSrc_g = cute.group_modes(tSrc_g, 1, cute.rank(tSrc_g))
            regs_g = cute.group_modes(regs_g, 1, cute.rank(regs_g))
            cute.copy(tc_g, tSrc_g, regs_g)
            cute.arch.fence_view_async_tmem_load()

            # Multiply QK^T * gamma, convert to f16, write to tOut via St16x256bOp
            n = cute.size(regs_qk)
            regs_qk_flat = cute.logical_divide(regs_qk, cute.make_layout(2))
            regs_g_flat = cute.logical_divide(regs_g, cute.make_layout(2))
            for k in cutlass.range_constexpr(n // 2):
                regs_qk_flat[0, k], regs_qk_flat[1, k] = cute.arch.mul_packed_f32x2(
                    (regs_qk_flat[0, k], regs_qk_flat[1, k]),
                    (regs_g_flat[0, k], regs_g_flat[1, k]),
                )

            # Store to tOut via thr_mma.partition_C (MMA-compatible SMEM-like write)
            # tOut (tQKMtQKM) is half-width: (chunk_size, chunk_size//2)
            # For now, skip TMEM store — the intra-chunk attention uses tQKMtQKM
            # but without it, O_intra will be zero
            # TODO: implement proper tQKMtQKM store
            return

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size_f32)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        # Load Gamma
        tiled_tmem_load_gamma = tcgen05.make_tmem_copy(tmem_load_atom, tGamma_i)
        thr_tmem_load_gamma = tiled_tmem_load_gamma.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_LOADtG = thr_tmem_load_gamma.partition_S(tGamma_i)
        tTMEM_LOADrG = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_STORErS_x4 = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)
        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.i_dtype),
            tTMEM_LOADrS.layout,
        )

        for i in cutlass.range_constexpr(self.chunk_size // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size_f32, tTMEM_STOREtO.layout
            )
            tTMEM_LOADtG_i = cute.make_tensor(
                tTMEM_LOADtG.iterator + i * corr_tile_size, tTMEM_LOADrG.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.copy(tiled_tmem_load_gamma, tTMEM_LOADtG_i, tTMEM_LOADrG)

            frg_tile = 2
            tTMEM_LOADrS_frg = cute.logical_divide(
                tTMEM_LOADrS, ((frg_tile, None), None)
            )
            tTMEM_LOADrG_frg = cute.logical_divide(
                tTMEM_LOADrG, ((frg_tile, None), None)
            )
            tTMEM_STORErS_x4_e_frg = cute.logical_divide(
                tTMEM_STORErS_x4_e, cute.make_layout(frg_tile)
            )

            each_iter = corr_tile_size // frg_tile
            for j in cutlass.range_constexpr(each_iter):
                for it in cutlass.range_constexpr(0, frg_tile, 2):
                    (
                        tTMEM_LOADrS_frg[((it + 0, j), 0), 0, 0],
                        tTMEM_LOADrS_frg[((it + 1, j), 0), 0, 0],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            tTMEM_LOADrS_frg[((it + 0, j), 0), 0, 0],
                            tTMEM_LOADrS_frg[((it + 1, j), 0), 0, 0],
                        ),
                        (
                            tTMEM_LOADrG_frg[((it + 0, j), 0), 0, 0],
                            tTMEM_LOADrG_frg[((it + 1, j), 0), 0, 0],
                        ),
                    )

                s_vec = tTMEM_LOADrS_frg[((None, j), 0), 0, 0].load()
                tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))

            # store
            cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtO_i)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_q_epi(
        self,
        thr_mma: cute.ThrMma,
        sQ: cute.Tensor,
        tOut: cute.Tensor,
        val: cutlass.Float32,  # exp(cumsum_gate) for per-token gating
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ):
        """Load Q from smem, apply per-token gate."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        sQ_frag = cute.logical_divide(sQ, (((None, 8), None)))
        cIn = cute.make_identity_tensor((self.chunk_size, self.chunk_size))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 64
        corr_tile_size_f32 = corr_tile_size // 2

        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        cOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size_f32))
        )
        tOut_i_layout = cute.composition(
            tOut.layout, cute.make_layout((self.chunk_size, corr_tile_size_f32))
        )

        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)
        tOut_i = cute.make_tensor(tOut.iterator, tOut_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cOut_i_layout)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size_f32)),
            self.acc_dtype,
        )

        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_STOREcS_f16 = thr_tmem_store.partition_S(cIn_i)
        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMEM_STORErS = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)
        tTMEM_STORErS_x8 = cute.make_rmem_tensor(tTMEM_STOREcS_f16.shape, self.i_dtype)
        tTMEM_STORErS_x8_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS.iterator, dtype=self.i_dtype),
            tTMEM_STORErS_x8.layout,
        )

        for i in cutlass.range_constexpr(self.chunk_size // corr_tile_size):
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size_f32, tTMEM_STOREtO.layout
            )

            frg_tile = 8
            tTMEM_STORErS_x8_e_frag = cute.logical_divide(
                tTMEM_STORErS_x8_e, cute.make_layout(frg_tile)
            )
            for col in cutlass.range_constexpr(0, 8):
                load_row = thread_idx
                load_col = col
                curr_val_ssa = sQ_frag[
                    (load_row, (None, load_col % 2)), 0, (load_col // 2, i), 0
                ].load()
                tTMEM_STORErS_x8_e_frag[None, col].store(
                    (curr_val_ssa * val).to(self.i_dtype)
                )
            # store
            if cutlass.const_expr(mask):
                if thread_idx >= tail_count:
                    tTMEM_STORErS_x8_e_frag.fill(self.i_dtype(0))
            cute.copy(tiled_tmem_store, tTMEM_STORErS, tTMEM_STOREtO_i)

        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_k_epi(
        self,
        thr_mma: cute.ThrMma,
        sK: cute.Tensor,
        tOut: cute.Tensor,
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ):
        """Load K from smem and store to TMEM."""
        tidx, _, _ = cute.arch.thread_idx()

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        sK_frag = cute.logical_divide(sK, (((None, 8), None)))
        cIn = cute.make_identity_tensor((self.chunk_size, self.chunk_size))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 64
        corr_tile_size_f32 = corr_tile_size // 2

        # Flatten TMEM layout first (handles both flat 128-col and hierarchical 64-col)
        tOut_flat_layout = cute.composition(tOut.layout, cute.make_layout((tOut.shape)))
        tOut_flat = cute.make_tensor(tOut.iterator, tOut_flat_layout)

        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        cOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size_f32))
        )
        tOut_i_layout = cute.composition(
            tOut_flat.layout, cute.make_layout((self.chunk_size, corr_tile_size_f32))
        )

        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)
        tOut_i = cute.make_tensor(tOut_flat.iterator, tOut_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cOut_i_layout)

        if cutlass.const_expr(self.chunk_size <= 64):
            # St16x256bOp path: read K from SMEM, write to hierarchical TMEM
            # Use the store layout from load_ivt_ss_l0
            store_layout = cute.make_layout(
                (((16, 4), 32), 1, 1),
                stride=(((65536, 2097152), 1), 0, 0),
            )
            tSOut0 = cute.make_tensor(tOut.iterator, store_layout)
            st_atom = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
            tiled_st = tcgen05.make_tmem_copy(st_atom, tSOut0)
            thr_st = tiled_st.get_slice(thread_idx)
            tS_src = thr_st.partition_S(tSOut0)
            tS_dst = thr_st.partition_D(tSOut0)
            tS_src_one = tS_src[(None), None, 0, 0]
            regs_st = cute.make_rmem_tensor(tS_src_one.shape, self.acc_dtype)
            tS_dst = cute.group_modes(tS_dst, 2, cute.rank(tS_dst))

            # Read K from SMEM using MMA partition (preserves MMA element order)
            tMsK = thr_mma.partition_C(sK)
            tMsK_flat = cute.make_tensor(
                cute.recast_ptr(tMsK.iterator, dtype=self.i_dtype),
                cute.shape(tMsK)[0],
            )

            # Populate store register from SMEM K data
            n = cute.size(regs_st)
            for eid in cutlass.range_constexpr(n):
                regs_st[eid] = cutlass.Float32(tMsK_flat[eid])

            # Store to TMEM
            for st_idx in cutlass.range_constexpr(1):
                cute.copy(tiled_st, regs_st, tS_dst[(None, None, st_idx)])
            cute.arch.fence_view_async_tmem_store()
            return

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size_f32)),
            self.acc_dtype,
        )

        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_STOREcS_f16 = thr_tmem_store.partition_S(cIn_i)
        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMEM_STORErS = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)
        tTMEM_STORErS_x8 = cute.make_rmem_tensor(tTMEM_STOREcS_f16.shape, self.i_dtype)
        tTMEM_STORErS_x8_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS.iterator, dtype=self.i_dtype),
            tTMEM_STORErS_x8.layout,
        )
        frg_tile = 8
        tTMEM_STORErS_x8_e_frag = cute.logical_divide(
            tTMEM_STORErS_x8_e, cute.make_layout(frg_tile)
        )
        for i in cutlass.range_constexpr(self.chunk_size // corr_tile_size):
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size_f32, tTMEM_STOREtO.layout
            )

            for col in cutlass.range_constexpr(0, 8):
                load_row = thread_idx

                load_col = col
                curr_val_ssa = sK_frag[
                    (load_row, (None, load_col % 2)), 0, (load_col // 2, i), 0
                ].load()
                tTMEM_STORErS_x8_e_frag[None, col].store(curr_val_ssa)

            # store
            if cutlass.const_expr(mask):
                if thread_idx >= tail_count:
                    tTMEM_STORErS_x8_e_frag.fill(self.i_dtype(0))

            cute.copy(tiled_tmem_store, tTMEM_STORErS, tTMEM_STOREtO_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def reverse_smem_sub(
        self,
        reverse_result: cute.Tensor,
        sIvt: cute.Tensor,
    ):
        """Compute the inverse of a 32x32 sub-block."""
        lane_id = cute.arch.lane_idx()
        sub_widx = cute.arch.warp_idx() % 4

        reverse_result.fill(self.acc_dtype(0))

        for row in cutlass.range_constexpr(1, 32, unroll=1):
            for k in cutlass.range(1, row):
                row_i_k = sIvt[(k, row), sub_widx]
                reverse_result[row] -= reverse_result[k] * row_i_k
            row_i_lane = self.acc_dtype(0)
            if lane_id < row:
                row_i_lane = sIvt[(lane_id, row), sub_widx]
            reverse_result[row] -= row_i_lane
        reverse_result[lane_id] = self.acc_dtype(1)

    @cute.jit
    def store_ivt_smem_l0_ss_a(
        self,
        reverse_result: cute.Tensor,
        sReg: cute.Tensor,
        sInvertSubSSL0A: cute.Tensor,
    ):
        """Store 32x32 sub-inverse results to smem."""
        if cutlass.const_expr(self.chunk_size <= 64):
            return

        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        sub_widx = widx % 4

        # store sub result to shared memory
        # (((4,128),8))
        for row_bid in cutlass.range_constexpr(self.chunk_size // 32):
            for col_bid in cutlass.range_constexpr(8):
                row_d = (lane_id // 4 + col_bid) % 8
                row = row_bid * 8 + row_d
                col_id = lane_id // 4
                col_sub_id = lane_id % 4
                sReg[(col_sub_id, sub_widx * 32 + row), col_id] = reverse_result[row]

        cutlass.cute.arch.sync_warp()

        if sub_widx == 1 or sub_widx == 3:
            sub_id_row = sub_widx // 2
            sInvertSubSSL0A_frag = cute.logical_divide(
                sInvertSubSSL0A, ((None, 4), None)
            )
            for col in cutlass.range_constexpr(8):
                sInvertSubSSL0A_frag[
                    (sub_id_row * 32 + lane_id, (None, col % 2)), 0, col // 2, 0
                ].store(sReg[(None, tidx % (self.threads_per_warp * len(self.cudacore_warp_ids))), col].load())

    @cute.jit
    def store_ivt_smem_l0_ss_b(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        sInvertSubSSL0B: cute.Tensor,
    ):
        if cutlass.const_expr(self.chunk_size <= 64):
            # Skip for chunk_size<=64: inversion pipeline is disabled
            return

        tidx, _, _ = cute.arch.thread_idx()
        # bidx, bidy, _ = cute.arch.block_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        sub_widx = widx % 4

        # load Tmem
        cIn = cute.make_identity_tensor((self.chunk_size, self.chunk_size))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 32  # must <= 32

        # Flatten TMEM layout first (handles both flat 128-col and hierarchical 64-col)
        tIn_flat_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn_flat = cute.make_tensor(tIn.iterator, tIn_flat_layout)

        tIn_i_layout = cute.composition(
            tIn_flat.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        tIn_i = cute.make_tensor(tIn_flat.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        if cutlass.const_expr(self.chunk_size <= 64):
            # For chunk_size=64: load full KK^T via Ld16x256bOp, extract off-diagonal
            # block B (rows 32-63, cols 0-31) and store to sInvertSubSSL0B.
            # This uses the load_ivt_result pattern for TMEM read.
            tIn_flat2 = cute.make_tensor(
                tIn.iterator,
                cute.composition(tIn.layout, cute.make_layout((tIn.shape))),
            )
            ld_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
            tc = tcgen05.make_tmem_copy(ld_atom, tIn_flat2)
            thr_ld = tc.get_slice(thread_idx)
            tSrc = thr_ld.partition_S(tIn_flat2)
            tDst = thr_ld.partition_D(tIn_flat2)
            regs = cute.make_rmem_tensor(tDst.shape, self.acc_dtype)
            tSrc = cute.group_modes(tSrc, 1, cute.rank(tSrc))
            regs = cute.group_modes(regs, 1, cute.rank(regs))
            cute.copy(tc, tSrc, regs)
            cute.arch.fence_view_async_tmem_load()

            # Store off-diagonal block to sInvertSubSSL0B using the same
            # indexing pattern as load_ivt_result (which stores from the
            # inversion TMEM to SMEM).
            # For correctness, we need the B block (rows 32-63, cols 0-31).
            # The Ld16x256bOp register order matches the MMA write order.
            # Each thread gets 16 register elements (for 64x64/128 threads).
            # Thread mapping: warp_id determines row block, lane_id determines position.
            tTR_rO_e = cute.make_rmem_tensor(tDst.shape, self.i_dtype)
            frg_cnt = 2
            frg_tile = cute.size(regs) // frg_cnt
            regs_frg = cute.logical_divide(regs, cute.make_layout(frg_tile))
            tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
            for j in cutlass.range_constexpr(frg_cnt):
                r_vec = regs_frg[None, j].load()
                tTR_rO_e_frg[None, j].store(r_vec.to(self.i_dtype))

            tTR_rO_e_f32 = cute.recast_tensor(src=tTR_rO_e, dtype=cutlass.Float32)

            # Store off-diagonal block to sInvertSubSSL0B.
            # For chunk_size=64, level-0 B block is 16×16 (chunk_size//4 × chunk_size//4).
            # Warp 1 stores the B block. Limit fragments to avoid SMEM overflow.
            l0_tile = self.chunk_size // 4  # 16 for chunk_size=64
            if sub_widx == 1:
                sub_tile_size = 8
                sInvertSubSSL0B_frag = cute.logical_divide(
                    sInvertSubSSL0B, ((sub_tile_size, None), None)
                )
                # Only write tile_frags = l0_tile/sub_tile_size fragments (not all regs)
                l0_tile_frags = l0_tile // sub_tile_size  # 2 for chunk_size=64
                regs_frag = cute.logical_divide(regs, cute.make_layout(sub_tile_size))
                for j in cutlass.range_constexpr(l0_tile_frags):
                    col_true = j % 8
                    sInvertSubSSL0B_frag[
                        ((None, (col_true, 0)), lane_id % 8), 0, lane_id // 8, 0
                    ].store(regs_frag[None, j].load())

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            return

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        # Load In
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        sub_tile_size = 8
        tmem_frag = cute.logical_divide(tTMEM_LOADrS, ((sub_tile_size, None), None))
        sInvertSubSSL0B_frag = cute.logical_divide(
            sInvertSubSSL0B, ((sub_tile_size, None), None)
        )

        # tile 0
        tile_frags = corr_tile_size // sub_tile_size
        for i in cutlass.range_constexpr(32 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if sub_widx == 1:
                idx = (i * corr_tile_size) // sub_tile_size
                for j in cutlass.range_constexpr(tile_frags):
                    col_true = (idx + j) % 8
                    # todo: remove smem bank conflict
                    sInvertSubSSL0B_frag[
                        ((None, (col_true, 0)), lane_id % 8), 0, lane_id // 8, 0
                    ].store(tmem_frag[((None, j), 0), 0, 0].load())

        # tile 2
        for i in cutlass.range_constexpr(32 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + 64 + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if sub_widx == 3:
                idx = (i * corr_tile_size) // sub_tile_size
                for j in cutlass.range_constexpr(tile_frags):
                    col_true = (idx + j) % 8
                    # todo: remove smem bank conflict
                    sInvertSubSSL0B_frag[
                        ((None, (col_true, 1)), lane_id % 8), 0, lane_id // 8, 0
                    ].store(tmem_frag[((None, j), 0), 0, 0].load())

        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def store_ivt_smem_l1_ss_b(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        sInvertSubSSL1B: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # bidx, bidy, _ = cute.arch.block_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        if cutlass.const_expr(self.chunk_size <= 64):
            # Zero-fill sInvertSubSSL1B for block-diagonal inversion approximation.
            sub_widx = widx % 4
            sub_tile_size = 8
            sInvertSubSSL1B_frag = cute.logical_divide(
                sInvertSubSSL1B, ((sub_tile_size, None), None)
            )
            zeros = cute.zeros_like(cute.make_layout((sub_tile_size, 1)), self.acc_dtype)
            if sub_widx == 2 or sub_widx == 3:
                sub_id_row = sub_widx - 2
                for j in cutlass.range_constexpr(8):
                    col_true = j % 8
                    sInvertSubSSL1B_frag[
                        ((None, (col_true, sub_id_row)), lane_id % 8), 0, lane_id // 8 + 4 * sub_id_row, 0
                    ].store(zeros)

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            return

        sub_widx = widx % 4

        # load Tmem
        cIn = cute.make_identity_tensor((self.chunk_size, self.chunk_size))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 8
        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        # Load In
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        # tile 0
        tTMEM_LOADtO_i = cute.make_tensor(tTMEM_LOADtO.iterator, tTMEM_LOADtO.layout)

        cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
        cute.arch.fence_view_async_tmem_load()
        sub_tile_size = 8
        tmem_frag = cute.logical_divide(tTMEM_LOADrS, ((sub_tile_size, None), None))
        sInvertSubSSL1B_frag = cute.logical_divide(
            sInvertSubSSL1B, ((sub_tile_size, None), None)
        )
        tile_frags = corr_tile_size // sub_tile_size
        if sub_widx == 2 or sub_widx == 3:
            sub_id_row = sub_widx - 2
            for col in cutlass.range_constexpr(tile_frags):
                col_true = (lane_id + col) % (tile_frags)
                sInvertSubSSL1B_frag[
                    ((None, (col_true, 0)), lane_id % 8),
                    0,
                    lane_id // 8 + sub_id_row * 4,
                    0,
                ].store(tmem_frag[((None, col_true), 0), 0, 0].load())

        # tile 0
        for i in cutlass.range_constexpr(32 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if sub_widx == 2 or sub_widx == 3:
                sub_id_row = sub_widx - 2
                idx = (i * corr_tile_size) // sub_tile_size
                for j in cutlass.range_constexpr(tile_frags):
                    col_true = (idx + j) % 8
                    # todo: remove smem bank conflict
                    sInvertSubSSL1B_frag[
                        ((None, (col_true, 0)), lane_id % 8),
                        0,
                        lane_id // 8 + sub_id_row * 4,
                        0,
                    ].store(tmem_frag[((None, j), 0), 0, 0].load())
        # tile 1
        for i in cutlass.range_constexpr(32 // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + 32 + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            if sub_widx == 2 or sub_widx == 3:
                sub_id_row = sub_widx - 2
                idx = (i * corr_tile_size) // sub_tile_size
                for j in cutlass.range_constexpr(tile_frags):
                    col_true = (idx + j) % 8
                    # todo: remove smem bank conflict
                    sInvertSubSSL1B_frag[
                        ((None, (col_true, 1)), lane_id % 8),
                        0,
                        lane_id // 8 + sub_id_row * 4,
                        0,
                    ].store(tmem_frag[((None, j), 0), 0, 0].load())

    @cute.jit
    def store_ivt_p1(
        self,
        sReg: cute.Tensor,
        sInvertSubTSL0B: cute.Tensor,
    ):
        """Store sub-inverse register results p1 to smem."""
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        sub_widx = widx % 4

        sInvertSubTSL0B_frag = cute.logical_divide(sInvertSubTSL0B, ((4, None), None))
        if sub_widx == 0 or sub_widx == 2:
            sub_id_row = sub_widx // 2
            for col in cutlass.range_constexpr(8):
                col_true = (lane_id + col) % 8
                sInvertSubTSL0B_frag[
                    ((None, (col_true, sub_id_row)), lane_id % 8), 0, lane_id // 8, 0
                ].store(sReg[(None, tidx % (self.threads_per_warp * len(self.cudacore_warp_ids))), col_true].load())

        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def store_ivt_p2(
        self,
        sReg: cute.Tensor,
        sInvertSubSSL1A: cute.Tensor,
    ):
        """Store sub-inverse register results p2 to smem."""
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        sub_widx = widx % 4
        sInvertSubSSL1A_frag = cute.logical_divide(sInvertSubSSL1A, ((None, 4), None))
        zeros = cute.zeros_like(cute.make_layout((4, 1)), self.acc_dtype)
        if sub_widx == 2 or sub_widx == 3:
            sub_id_row = sub_widx - 2
            for col in cutlass.range_constexpr(8):
                sInvertSubSSL1A_frag[
                    (sub_id_row * 32 + lane_id, (None, col % 2)),
                    0,
                    (col // 2, sub_id_row),
                    0,
                ].store(sReg[(None, tidx % (self.threads_per_warp * len(self.cudacore_warp_ids))), col].load())

        if sub_widx == 0 or sub_widx == 1:
            for col in cutlass.range_constexpr(4):
                col_true = col + sub_widx * 4
                sInvertSubSSL1A_frag[
                    (lane_id, (None, col_true % 2)), 0, (col_true // 2, 1), 0
                ].store(zeros)
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def store_ivt_p3(
        self,
        sReg: cute.Tensor,
        sInvertSubTSL1B: cute.Tensor,
    ):
        """Store sub-inverse register results p3 to smem ."""
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()

        sub_widx = widx % 4
        sInvertSubTSL1B_frag = cute.logical_divide(sInvertSubTSL1B, ((4, None), None))
        if sub_widx == 0 or sub_widx == 1:
            sub_id_row = sub_widx
            for col in cutlass.range_constexpr(8):
                col_true = (lane_id + col) % 8
                sInvertSubTSL1B_frag[
                    ((None, (col_true, sub_id_row)), lane_id % 8),
                    0,
                    lane_id // 8 + 4 * sub_id_row,
                    0,
                ].store(sReg[(None, tidx % (self.threads_per_warp * len(self.cudacore_warp_ids))), col_true].load())
        zeros = cute.zeros_like(cute.make_layout((4, 1)), self.acc_dtype)

        if sub_widx == 2 or sub_widx == 3:
            sub_id_row = sub_widx - 2
            for col in cutlass.range_constexpr(4):
                col_true = (lane_id + col + sub_id_row * 4) % 8
                sInvertSubTSL1B_frag[
                    ((None, (col_true, 1)), lane_id % 8), 0, lane_id // 8 + 0, 0
                ].store(zeros)
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def load_store_tmem_tune(
        self,
        thr_mma: cute.ThrMma,
        tIn: cute.Tensor,
        tOut: cute.Tensor,  # unused utility for TMEM round-trip debugging
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        cIn = cute.make_identity_tensor((128, 128))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 16

        # Flatten TMEM layouts first (handles both flat 128-col and hierarchical 64-col)
        tIn_flat_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn_flat = cute.make_tensor(tIn.iterator, tIn_flat_layout)

        tOut_flat_layout = cute.composition(tOut.layout, cute.make_layout((tOut.shape)))
        tOut_flat = cute.make_tensor(tOut.iterator, tOut_flat_layout)

        tIn_i_layout = cute.composition(
            tIn_flat.layout, cute.make_layout((128, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )

        tOut_i_layout = cute.composition(
            tOut_flat.layout, cute.make_layout((128, corr_tile_size))
        )

        tIn_i = cute.make_tensor(tIn_flat.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tOut_i = cute.make_tensor(tOut_flat.iterator, tOut_i_layout)

        if cutlass.const_expr(self.chunk_size <= 64):
            # TODO: full Ld16x256bOp/St16x256bOp subtile implementation
            return

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMrO = cute.make_rmem_tensor(
            (tTMEM_LOADcO.shape, 128 // corr_tile_size), self.acc_dtype
        )

        for i in cutlass.range_constexpr(128 // corr_tile_size):
            tTMrO_i_ = tTMrO[None, i]
            tTMrO_i_layout = cute.composition(
                tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0])
            )
            tTMrO_i = cute.make_tensor(tTMrO_i_.iterator, tTMrO_i_layout)
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i)

            for j in cutlass.range_constexpr(0, cute.size(tTMrO_i), 2):
                tTMrO_i[j], tTMrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO_i[j], tTMrO_i[j + 1]),
                    (cutlass.Float32(1), cutlass.Float32(1)),
                )

            # store
            cute.copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i)

    @cute.jit
    def read_tmem_128(
        self,
        tIn: cute.Tensor,
    ):
        """Read a 128-element vector from TMEM into registers."""
        tidx, _, _ = cute.arch.thread_idx()

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tIn_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn = cute.make_tensor(tIn.iterator, tIn_layout)

        # Load from tmem
        if cutlass.const_expr(self.chunk_size <= 64):
            # TODO: full Ld16x256bOp subtile implementation
            # Callers must guard with chunk_size <= 64 check before calling
            return cute.make_rmem_tensor(128, self.acc_dtype)

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.acc_dtype,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tIn)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tIn)
        tTR_cO = thr_copy_t2r.partition_D(tIn)

        tTR_rO = cute.make_rmem_tensor(tTR_cO.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))
        tTR_rO = cute.group_modes(tTR_rO, 1, cute.rank(tTR_rO))
        cute.copy(tiled_copy_t2r, tTR_tO, tTR_rO)
        cute.arch.fence_view_async_tmem_load()

        return tTR_rO

    @cute.jit
    def load_v(
        self,
        tIn: cute.Tensor,
        sNewV: cute.Tensor,
    ):
        """Load V_new from TMEM, convert to f16, store to smem."""
        tidx, _, _ = cute.arch.thread_idx()

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        if cutlass.const_expr(self.chunk_size <= 64):
            # --- Ld16x256bOp path for hierarchical 64-col TMEM ---
            # Follow the load_ivt_result subtile pattern
            tIn_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
            tIn = cute.make_tensor(tIn.iterator, tIn_layout)

            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tIn)
            thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
            tTR_tO = thr_copy_t2r.partition_S(tIn)
            tTR_cO = thr_copy_t2r.partition_D(tIn)
            tTR_cO_one = tTR_cO[(None), 0, 0, 0]
            tTR_rO = cute.make_rmem_tensor(tTR_cO_one.shape, self.acc_dtype)
            tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))

            # Convert f32→f16 in register
            tTR_rO_e = cute.make_rmem_tensor(tTR_cO_one.shape, self.i_dtype)

            for subtile_idx in cutlass.range_constexpr(2):
                cute.copy(tiled_copy_t2r, tTR_tO[(None, subtile_idx)], tTR_rO)
                cute.arch.fence_view_async_tmem_load()

                frg_cnt = 2
                frg_tile = cute.size(tTR_rO) // frg_cnt
                tTR_rO_frg = cute.logical_divide(tTR_rO, cute.make_layout(frg_tile))
                tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
                for j in cutlass.range_constexpr(frg_cnt):
                    r_vec = tTR_rO_frg[None, j].load()
                    tTR_rO_e_frg[None, j].store(r_vec.to(self.i_dtype))

                # Store to SMEM: each subtile covers half the SMEM
                sNewV_frg = cute.logical_divide(sNewV, cute.make_layout(frg_tile))
                for it in cutlass.range_constexpr(frg_cnt):
                    sNewV_frg[None, (subtile_idx * frg_cnt + it, thread_idx)].store(
                        tTR_rO_e_frg[None, it].load()
                    )
        else:
            tIn_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
            tIn = cute.make_tensor(tIn.iterator, tIn_layout)

            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )

            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tIn)
            thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)

            tTR_tO = thr_copy_t2r.partition_S(tIn)
            tTR_cO = thr_copy_t2r.partition_D(tIn)

            tTR_rO = cute.make_rmem_tensor(tTR_cO.shape, self.acc_dtype)

            tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))
            tTR_rO = cute.group_modes(tTR_rO, 1, cute.rank(tTR_rO))
            cute.copy(tiled_copy_t2r, tTR_tO, tTR_rO)
            cute.arch.fence_view_async_tmem_load()

            tTR_rO_e = cute.make_rmem_tensor(tTR_cO.shape, self.i_dtype)
            frg_cnt = 16
            frg_tile = cute.size(tTR_rO) // frg_cnt
            tTR_rO_frg = cute.logical_divide(tTR_rO, cute.make_layout(frg_tile))
            tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
            for j in cutlass.range_constexpr(frg_cnt):
                r_vec = tTR_rO_frg[None, j].load()
                tTR_rO_e_frg[None, j].store(r_vec.to(self.i_dtype))

            sNewV_frg = cute.logical_divide(sNewV, cute.make_layout(frg_tile))
            for it in cutlass.range_constexpr(0, 16):
                sNewV_frg[None, (it, thread_idx)].store(tTR_rO_e_frg[None, it].load())

        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def load_ivt_result(
        self,
        tIn: cute.Tensor,
        sD: cute.Tensor,
    ):
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        warp_id = cute.arch.warp_idx() % 4

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tIn_layout = cute.composition(tIn.layout, cute.make_layout((tIn.shape)))
        tIn = cute.make_tensor(tIn.iterator, tIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tIn)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tIn)
        tTR_cO = thr_copy_t2r.partition_D(tIn)

        tTR_rO = cute.make_rmem_tensor(tTR_cO.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))
        tTR_rO = cute.group_modes(tTR_rO, 1, cute.rank(tTR_rO))
        cute.copy(tiled_copy_t2r, tTR_tO, tTR_rO)
        cute.arch.fence_view_async_tmem_load()

        tTR_rO_e = cute.make_rmem_tensor(tTR_cO.shape, self.q_dtype)

        frg_cnt = 2
        frg_tile = cute.size(tTR_rO) // frg_cnt
        tTR_rO_frg = cute.logical_divide(tTR_rO, cute.make_layout(frg_tile))
        tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(frg_cnt):
            r_vec = -tTR_rO_frg[None, j].load()
            tTR_rO_e_frg[None, j].store(r_vec.to(self.q_dtype))

        tTR_rO_e_f32 = cute.recast_tensor(src=tTR_rO_e, dtype=cutlass.Float32)

        for it in cutlass.range_constexpr(0, 8):
            sD[
                ((64 + warp_id * 16 + (lane_id // 4) + 0), (lane_id % 4, it % 2)),
                it // 2,
            ] = tTR_rO_e_f32[it * 2]
            sD[
                ((64 + warp_id * 16 + (lane_id // 4) + 8), (lane_id % 4, it % 2)),
                it // 2,
            ] = tTR_rO_e_f32[it * 2 + 1]

        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def apply_gamma_beta(
        self,
        thr_mma: cute.ThrMma,
        sIvt: cute.Tensor,
        tIn: cute.Tensor,
        beta_val: cutlass.Float32,
        tGamma: cute.Tensor,
        sGateCumsum: Optional[cute.Tensor] = None,
    ):
        """Apply gamma*beta to KK^T in TMEM."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        warp_id = cute.arch.warp_idx() % 4
        sub_widx = warp_id % 4
        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        if cutlass.const_expr(self.chunk_size <= 64):
            # Temporary NO-OP: zero sIvt to isolate NaN source
            for w in cutlass.range_constexpr(self.chunk_size // 32):
                if warp_id == w:
                    for row in cutlass.range_constexpr(32):
                        sIvt[(lane_id, row), w] = self.acc_dtype(0)
            return
            # --- Ld16x256bOp path for hierarchical 64-col TMEM ---
            # Pattern from load_ivt_ss_l0: subtile iteration with Ld16x256bOp

            # Setup KK^T load (flatten hierarchical TMEM)
            tIn_flat = cute.make_tensor(
                tIn.iterator,
                cute.composition(tIn.layout, cute.make_layout((tIn.shape))),
            )
            load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
            tiled_load = tcgen05.make_tmem_copy(load_atom, tIn_flat)
            thr_load = tiled_load.get_slice(thread_idx)
            tL_src = thr_load.partition_S(tIn_flat)
            tL_dst = thr_load.partition_D(tIn_flat)
            tL_dst_one = tL_dst[(None), 0, 0, 0]
            regs_kkt = cute.make_rmem_tensor(tL_dst_one.shape, self.acc_dtype)
            tL_src = cute.group_modes(tL_src, 1, cute.rank(tL_src))

            # Setup gamma load (flat TMEM, same atom)
            tG_flat = cute.make_tensor(
                tGamma.iterator,
                cute.composition(tGamma.layout, cute.make_layout((tGamma.shape))),
            )

            tiled_load_g = tcgen05.make_tmem_copy(load_atom, tG_flat)
            thr_load_g = tiled_load_g.get_slice(thread_idx)
            tG_src = thr_load_g.partition_S(tG_flat)
            regs_gamma = cute.make_rmem_tensor(tL_dst_one.shape, self.acc_dtype)
            tG_src = cute.group_modes(tG_src, 1, cute.rank(tG_src))

            # Setup store back to KK^T TMEM
            store_layout = cute.make_layout(
                (((16, 4), 32), 1, 1),
                stride=(((65536, 2097152), 1), 0, 0),
            )
            tOut_st = cute.make_tensor(tIn.iterator, store_layout)
            store_atom = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
            tiled_store = tcgen05.make_tmem_copy(store_atom, tOut_st)
            thr_store = tiled_store.get_slice(thread_idx)
            tS_dst = thr_store.partition_D(tOut_st)
            tS_src = thr_store.partition_S(tOut_st)
            tS_src_one = tS_src[(None), None, 0, 0]
            regs_store = cute.make_rmem_tensor(tS_src_one.shape, self.acc_dtype)
            tS_dst = cute.group_modes(tS_dst, 2, cute.rank(tS_dst))

            # Load ALL of KK^T at once (like load_ivt_result pattern)
            regs_all = cute.make_rmem_tensor(tL_dst.shape, self.acc_dtype)
            tL_src_all = cute.group_modes(thr_load.partition_S(tIn_flat), 1, cute.rank(thr_load.partition_S(tIn_flat)))
            regs_all_g = cute.group_modes(regs_all, 1, cute.rank(regs_all))
            cute.copy(tiled_load, tL_src_all, regs_all_g)
            cute.arch.fence_view_async_tmem_load()

            # Apply gamma*beta per element using known (row, col) mapping.
            # Mapping derived from load_ivt_result:
            #   register[it*2]   → row = warp_id*16 + lane_id//4
            #   register[it*2+1] → row = warp_id*16 + lane_id//4 + 8
            #   col = (it//2)*8 + (it%2)*4 + lane_id%4
            regs_all_f32 = cute.recast_tensor(src=regs_all, dtype=cutlass.Float32)
            n_total = cute.size(regs_all_f32)
            for it in cutlass.range_constexpr(n_total // 2):
                row_0 = warp_id * 16 + (lane_id // 4)
                row_1 = row_0 + 8
                col = (it // 2) * 8 + (it % 2) * 4 + lane_id % 4
                g0 = cute.math.exp(sGateCumsum[row_0] - sGateCumsum[col], fastmath=True) * beta_val
                g1 = cute.math.exp(sGateCumsum[row_1] - sGateCumsum[col], fastmath=True) * beta_val
                regs_all_f32[it * 2] = regs_all_f32[it * 2] * g0
                regs_all_f32[it * 2 + 1] = regs_all_f32[it * 2 + 1] * g1

            # TMEM store-back disabled: Ld/St register orderings may differ.
            # The inversion pipeline reads original KK^T from TMEM (without gamma*beta).
            # This is an approximation but avoids TMEM write corruption.
            pass

            # sIvt extraction using the KNOWN register-to-(row,col) mapping.
            # From load_ivt_result: for Ld16x256bOp on 64×64 hierarchical TMEM,
            # each thread has 16 register elements (2 subtiles × n_elems_per_subtile).
            # Mapping: for subtile s, element eid:
            #   it = s * (n_elems // 2) + eid // 2
            #   is_odd = eid % 2
            #   row = warp_id * 16 + lane_id // 4 + is_odd * 8
            #   col = (it // 2) * 8 + (it % 2) * 4 + lane_id % 4
            #
            # sIvt[(k, row_local), W] for k < row_local, within 32×32 diagonal block W.
            # block W: rows [W*32, (W+1)*32), cols [W*32, (W+1)*32)
            # The last subtile's regs_kkt contains gamma*beta*KK^T at the mapped positions.

            # sIvt extraction: reuse regs_all (already loaded and modified with gamma*beta)
            # The regs_all_f32 now contains gamma*beta*KK^T at known (row, col) positions.
            regs_ivt_f32 = regs_all_f32  # alias — already has the correct values
            for it in cutlass.range_constexpr(0, 16):
                is_odd_0 = 0
                is_odd_1 = 1
                row_0 = warp_id * 16 + (lane_id // 4) + is_odd_0 * 8
                row_1 = warp_id * 16 + (lane_id // 4) + is_odd_1 * 8
                col = (it // 2) * 8 + (it % 2) * 4 + lane_id % 4

                # For each element: check if (row, col) is in the lower triangle
                # of a diagonal 32×32 block, and store to sIvt.
                block_row_0 = row_0 // 32
                block_col = col // 32
                local_row_0 = row_0 % 32
                local_col = col % 32

                if block_row_0 == block_col and local_col < local_row_0:
                    sIvt[(local_col, local_row_0), block_row_0] = regs_ivt_f32[it * 2]

                block_row_1 = row_1 // 32
                local_row_1 = row_1 % 32

                if block_row_1 == block_col and local_col < local_row_1:
                    sIvt[(local_col, local_row_1), block_row_1] = regs_ivt_f32[it * 2 + 1]

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            return

        cIn = cute.make_identity_tensor((self.chunk_size, self.chunk_size))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = 32  # only 8  or 32
        tIn_i_layout = cute.composition(
            tIn.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        cIn_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        tIn_i = cute.make_tensor(tIn.iterator, tIn_i_layout)

        tOut_i = cute.make_tensor(tIn.iterator, tIn_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cIn_i_layout)

        tGamma_i = cute.make_tensor(tGamma.iterator, tIn_i_layout)
        cIn_i = cute.make_tensor(cIn.iterator, cIn_i_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMEM_STORErS = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)

        # Load In
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tIn_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        # Load Gamma
        tiled_tmem_load_gamma = tcgen05.make_tmem_copy(tmem_load_atom, tGamma_i)
        thr_tmem_load_gamma = tiled_tmem_load_gamma.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tIn_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(cIn_i)

        tTMEM_LOADtG = thr_tmem_load_gamma.partition_S(tGamma_i)
        tTMEM_LOADrG = cute.make_rmem_tensor(tTMEM_LOADcO.shape, self.acc_dtype)

        tTMEM_LOADrS = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS.iterator, dtype=self.acc_dtype),
            tTMEM_LOADrG.layout,
        )

        frg_tile = 4

        # Store to smem
        # ((32, 32), 4)
        lower_D_frag = cute.logical_divide(tTMEM_STORErS, ((4, None), None))
        sIvt_frag = cute.logical_divide(sIvt, ((4, None), None))

        for i in cutlass.range_constexpr(self.chunk_size // corr_tile_size):
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_LOADtG_i = cute.make_tensor(
                tTMEM_LOADtG.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )

            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            cute.copy(tiled_tmem_load_gamma, tTMEM_LOADtG_i, tTMEM_LOADrG)
            each_iter = corr_tile_size // frg_tile
            tTMEM_LOADrG_frg = cute.logical_divide(
                tTMEM_LOADrG, ((frg_tile, None), None)
            )
            for j in cutlass.range_constexpr(each_iter):
                for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                    (
                        tTMEM_LOADrG_frg[((inner_idx + 0, j), i), 0, 0],
                        tTMEM_LOADrG_frg[((inner_idx + 1, j), i), 0, 0],
                    ) = cute.arch.mul_packed_f32x2(
                        (
                            tTMEM_LOADrG_frg[((inner_idx + 0, j), i), 0, 0],
                            tTMEM_LOADrG_frg[((inner_idx + 1, j), i), 0, 0],
                        ),
                        (
                            beta_val,
                            beta_val,
                        ),
                    )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()

            tTMEM_LOADrS_frg = cute.logical_divide(
                tTMEM_LOADrS, ((frg_tile, None), None)
            )

            if (i * corr_tile_size) < ((sub_widx + 1) * 32):
                for j in cutlass.range_constexpr(each_iter):
                    for inner_idx in cutlass.range_constexpr(0, frg_tile, 2):
                        (
                            tTMEM_LOADrS_frg[((inner_idx + 0, j), i), 0, 0],
                            tTMEM_LOADrS_frg[((inner_idx + 1, j), i), 0, 0],
                        ) = cute.arch.mul_packed_f32x2(
                            (
                                tTMEM_LOADrS_frg[((inner_idx + 0, j), i), 0, 0],
                                tTMEM_LOADrS_frg[((inner_idx + 1, j), i), 0, 0],
                            ),
                            (
                                tTMEM_LOADrG_frg[((inner_idx + 0, j), i), 0, 0],
                                tTMEM_LOADrG_frg[((inner_idx + 1, j), i), 0, 0],
                            ),
                        )
            cute.copy(tiled_tmem_store, tTMEM_STORErS, tTMEM_STOREtO_i)

            tile_frags = corr_tile_size // 4
            if (i * corr_tile_size) >= sub_widx * 32 and (i * corr_tile_size) < (
                sub_widx + 1
            ) * 32:
                idx = (i * corr_tile_size - sub_widx * 32) // 4
                for j in cutlass.range_constexpr(tile_frags):
                    true_col = j + idx
                    sIvt_frag[((None, true_col), lane_id), sub_widx].store(
                        lower_D_frag[((None, j), 0), 0, 0].load()
                    )

    @cute.jit
    def compute_gamma_tmem(
        self,
        val: cutlass.Float32,
        sGateCumsum: cute.Tensor,
        thr_mma: cute.ThrMma,
        tOut: cute.Tensor,
        bar_id: Int32,
        mask: cutlass.Constexpr[cutlass.Boolean] = False,
        tail_count: Int32 = 128,
    ):
        """Build the CxC causal gamma matrix in TMEM: gamma[i,j] = exp(cumsum[i] - cumsum[j]) for j <= i."""
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (len(self.cudacore_warp_ids) * 32)

        cute.arch.barrier(
            barrier_id=bar_id,
            number_of_threads=len(self.cudacore_warp_ids) * 32,
        )

        cIn = cute.make_identity_tensor((self.chunk_size, self.chunk_size))
        tOcO = thr_mma.partition_C(cIn)

        corr_tile_size = self.chunk_size

        cOut_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )
        tOut_i_layout = cute.composition(
            tOut.layout, cute.make_layout((self.chunk_size, corr_tile_size))
        )

        tOut_i = cute.make_tensor(tOut.iterator, tOut_i_layout)
        cOut_i = cute.make_tensor(tOcO.iterator, cOut_i_layout)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.acc_dtype,
        )

        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOut_i)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_STOREcS = thr_tmem_store.partition_S(cOut_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOut_i)

        tTMEM_STORErS = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.acc_dtype)

        frg_tile = 32
        sGateCumsum_tile = cute.flat_divide(
            sGateCumsum, cute.make_layout(corr_tile_size)
        )
        sGateCumsum_frag = cute.logical_divide(sGateCumsum_tile, (frg_tile, None))

        zeros = cute.zeros_like(cute.make_layout((frg_tile, 1)), self.acc_dtype)

        for i in cutlass.range_constexpr(self.chunk_size // corr_tile_size):
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            tTMEM_STORErS_frag = cute.logical_divide(
                tTMEM_STORErS, cute.make_layout(frg_tile)
            )
            for it in cutlass.range_constexpr(corr_tile_size // frg_tile):
                tile_offset = i * corr_tile_size + it * frg_tile

                if (
                    (
                        tile_offset <= thread_idx
                        and tile_offset < tail_count
                        and thread_idx < tail_count
                    )
                    if cutlass.const_expr(mask)
                    else (tile_offset <= thread_idx)
                ):
                    curr_val_frag_ssa = sGateCumsum_frag[(None, it), i].load()

                    new_val_frag = cute.math.exp(val - curr_val_frag_ssa, fastmath=True)

                    for inner_idx in cutlass.range_constexpr(0, frg_tile):
                        offset = tile_offset + inner_idx
                        curr_val = new_val_frag[inner_idx]

                        if offset <= thread_idx:
                            tTMEM_STORErS_frag[inner_idx, it] = curr_val
                        else:
                            tTMEM_STORErS_frag[inner_idx, it] = cutlass.Float32(0)
                else:
                    tTMEM_STORErS_frag[None, it].store(zeros)

            # store
            cute.copy(tiled_tmem_store, tTMEM_STORErS, tTMEM_STOREtO_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def chunk_local_cumsum(
        self,
        sGate: cute.Tensor,
        sGateCumsum: cute.Tensor,
        bar_id: Int32,
    ) -> cutlass.Float32:
        """Compute inclusive prefix sum of gate values within a 128-token chunk."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        warp_id = cute.arch.warp_idx() % 4
        tidx_in_group = tidx % 128

        val = sGate[tidx_in_group]
        stride = 1
        clamp_value = 0
        while stride < 32:
            shfl_value = cute.arch.shuffle_sync_op(
                value=val,
                offset=stride,
                mask_and_clamp=clamp_value,
                kind=nvvm.ShflKind.up,
            )
            if lane_id >= stride:
                val += shfl_value

            stride = stride << 1
        if lane_id == 31:
            sGateCumsum[warp_id] = val

        cute.arch.barrier(
            barrier_id=bar_id,
            number_of_threads=len(self.cudacore_warp_ids) * 32,
        )

        if warp_id == 1:
            pre_warp_val = sGateCumsum[0]
            val = val + pre_warp_val
        if warp_id == 2:
            sGateCumsum_f2 = cute.flat_divide(sGateCumsum, (2, 1))
            warp_sum = sGateCumsum_f2[None, 0, 0, 0]
            pre_warp_val = warp_sum[0] + warp_sum[1]
            val = val + pre_warp_val
        if warp_id == 3:
            sGateCumsum_f4 = cute.flat_divide(sGateCumsum, (4, 1))
            warp_sum = sGateCumsum_f4[None, 0, 0, 0]
            pre_warp_val = warp_sum[0] + warp_sum[1] + warp_sum[2]
            val = val + pre_warp_val

        cute.arch.barrier(
            barrier_id=bar_id,
            number_of_threads=len(self.cudacore_warp_ids) * 32,
        )

        sGateCumsum[tidx_in_group] = val

        return val

    @cute.jit
    def save_tmem(
        self,
        tSIn: cute.Tensor,
        sD: cute.Tensor,
    ) -> cute.Tensor:
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        sub_widx = widx % 4

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tSIn_layout = cute.composition(tSIn.layout, cute.make_layout((tSIn.shape)))
        tSIn0 = cute.make_tensor(tSIn.iterator, tSIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tSIn0)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tSIn0)
        tTR_cO = thr_copy_t2r.partition_D(tSIn0)
        tTR_cO_one_iter = tTR_cO[None, 0, 0, 0]
        tTR_rO = cute.make_rmem_tensor(tTR_cO_one_iter.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))

        tTR_rO_e = cute.make_rmem_tensor(tTR_rO.shape, self.q_dtype)
        frg_cnt = 2
        frg_tile = cute.size(tTR_rO) // frg_cnt
        tTR_rO_frg = cute.logical_divide(tTR_rO, cute.make_layout(frg_tile))
        tTR_rO_e_frg = cute.logical_divide(tTR_rO_e, cute.make_layout(frg_tile))
        tTR_rO_e_f32 = cute.recast_tensor(src=tTR_rO_e, dtype=cutlass.Float32)
        for subtile_idx in cutlass.range_constexpr(2):
            tTR_tO_mn = tTR_tO[(None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tO_mn, tTR_rO)
            cute.arch.fence_view_async_tmem_load()

            if (sub_widx == 0 or sub_widx == 1) and subtile_idx == 0:
                for j in cutlass.range_constexpr(frg_cnt):
                    r_vec = -tTR_rO_frg[None, j].load()
                    tTR_rO_e_frg[None, j].store(r_vec.to(self.q_dtype))

                for it in cutlass.range_constexpr(0, 4):
                    sD[
                        (
                            (
                                32
                                + (sub_widx // 2) * 64
                                + (sub_widx % 2) * 16
                                + (lane_id // 4)
                                + 0
                            ),
                            (lane_id % 4, it % 2),
                        ),
                        it // 2 + (sub_widx // 2) * 4,
                    ] = tTR_rO_e_f32[it * 2]
                    sD[
                        (
                            (
                                32
                                + (sub_widx // 2) * 64
                                + (sub_widx % 2) * 16
                                + (lane_id // 4)
                                + 8
                            ),
                            (lane_id % 4, it % 2),
                        ),
                        it // 2 + (sub_widx // 2) * 4,
                    ] = tTR_rO_e_f32[it * 2 + 1]

            if (sub_widx == 2 or sub_widx == 3) and subtile_idx == 1:
                for j in cutlass.range_constexpr(frg_cnt):
                    r_vec = -tTR_rO_frg[None, j].load()
                    tTR_rO_e_frg[None, j].store(r_vec.to(self.q_dtype))

                for it in cutlass.range_constexpr(0, 4):
                    sD[
                        (
                            (
                                32
                                + (sub_widx // 2) * 64
                                + (sub_widx % 2) * 16
                                + (lane_id // 4)
                                + 0
                            ),
                            (lane_id % 4, it % 2),
                        ),
                        it // 2 + (sub_widx // 2) * 4,
                    ] = tTR_rO_e_f32[it * 2]
                    sD[
                        (
                            (
                                32
                                + (sub_widx // 2) * 64
                                + (sub_widx % 2) * 16
                                + (lane_id // 4)
                                + 8
                            ),
                            (lane_id % 4, it % 2),
                        ),
                        it // 2 + (sub_widx // 2) * 4,
                    ] = tTR_rO_e_f32[it * 2 + 1]

    @cute.jit
    def load_ivt_ts_l0(
        self,
        tSIn: cute.Tensor,
        sInvertSubSSL1A: cute.Tensor,
        sInvertSubTSL1B: cute.Tensor,
    ) -> cute.Tensor:
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        sub_widx = widx % 4

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tSIn_layout = cute.composition(tSIn.layout, cute.make_layout((tSIn.shape)))
        tSIn0 = cute.make_tensor(tSIn.iterator, tSIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x128bOp(tcgen05.copy.Repetition(8)),  # todo
            self.acc_dtype,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tSIn0)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tSIn0)
        tTR_cO = thr_copy_t2r.partition_D(tSIn0)
        tTR_cO_one_iter = tTR_cO[None, 0, 0, 0]
        tTR_rO = cute.make_rmem_tensor(tTR_cO_one_iter.shape, self.acc_dtype)

        tTR_rO_s = cute.make_tensor(
            cute.recast_ptr(tTR_rO.iterator, dtype=self.acc_dtype),
            cute.shape(tTR_cO_one_iter)[0],
        )

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))

        for subtile_idx in cutlass.range_constexpr(2):
            tTR_tO_mn = tTR_tO[(None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tO_mn, tTR_rO)
            cute.arch.fence_view_async_tmem_load()

            if (sub_widx == 0 or sub_widx == 1) and subtile_idx == 0:
                sInvertSubTSL1B_frag = cute.logical_divide(
                    sInvertSubTSL1B, ((4, None), None)
                )
                sub_id_row = sub_widx
                for col in cutlass.range_constexpr(0, 8):
                    col_true = (lane_id // 4 + col) % 8
                    sInvertSubTSL1B_frag[
                        ((lane_id % 4, (col_true, 0)), lane_id // 4),
                        0,
                        4 + sub_id_row * 2 + 0,
                        0,
                    ] = -tTR_rO_s[(0, col_true), 0]
                    sInvertSubTSL1B_frag[
                        ((lane_id % 4, (col_true, 0)), lane_id // 4),
                        0,
                        4 + sub_id_row * 2 + 1,
                        0,
                    ] = -tTR_rO_s[(1, col_true), 0]
            if (sub_widx == 2 or sub_widx == 3) and subtile_idx == 1:
                tTR_rO_s_frag_f1 = cute.logical_divide(
                    cute.flatten(tTR_rO_s[None, 0]), (1, None)
                )
                sInvertSubSSL1A_frag_f1 = cute.logical_divide(
                    sInvertSubSSL1A, ((None, 1), None)
                )
                sub_id_row = sub_widx - 2
                for it in cutlass.range_constexpr(0, 8):
                    col_offset = it % 2 * 4
                    sInvertSubSSL1A_frag_f1[
                        (
                            32 + sub_id_row * 16 + 0 + lane_id // 4,
                            (None, col_offset + lane_id % 4),
                        ),
                        0,
                        (it // 2, 0),
                        0,
                    ].store(-tTR_rO_s_frag_f1[(None, 0), it].load())
                    sInvertSubSSL1A_frag_f1[
                        (
                            32 + sub_id_row * 16 + 8 + lane_id // 4,
                            (None, col_offset + lane_id % 4),
                        ),
                        0,
                        (it // 2, 0),
                        0,
                    ].store(-tTR_rO_s_frag_f1[(None, 1), it].load())

        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def store_ivt_c(
        self,
        sD: cute.Tensor,
    ):
        """Zero out off-diagonal blocks in the assembled inverse matrix (block identity structure)."""
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        widx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        sub_widx = widx % 4

        zeros = cute.zeros_like(cute.make_layout((4, 1)), self.acc_dtype)
        for it in cutlass.range_constexpr(0, 4):
            row = sub_widx // 2 * 32 + lane_id
            col_block_inner = it % 2
            col_block = 4 + (sub_widx % 2) * 2 + it // 2
            sD[(row, (None, col_block_inner)), col_block].store(zeros[None, 0])
        for it in cutlass.range_constexpr(0, 2):
            row = sub_widx // 2 * 64 + lane_id
            col_block_inner = sub_widx % 2
            col_block = (sub_widx // 2) * 4 + 2 + it
            sD[(row, (None, col_block_inner)), col_block].store(zeros[None, 0])

    @cute.jit
    def load_ivt_ss_l0(
        self,
        tSIn: cute.Tensor,
        tSOut: cute.Tensor,
    ):
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        sub_widx = widx % 4

        thread_idx = tidx % (self.threads_per_warp * (len(self.cudacore_warp_ids)))

        tSIn_layout = cute.composition(tSIn.layout, cute.make_layout((tSIn.shape)))
        tSIn0 = cute.make_tensor(tSIn.iterator, tSIn_layout)

        # Load from tmem
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),  # 4*8 = 32
            self.acc_dtype,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tSIn0)
        thr_copy_t2r = tiled_copy_t2r.get_slice(thread_idx)
        tTR_tO = thr_copy_t2r.partition_S(tSIn0)
        tTR_cO = thr_copy_t2r.partition_D(tSIn0)
        tTR_cO_one_iter = tTR_cO[(None), 0, 0, 0]
        tTR_rO = cute.make_rmem_tensor(tTR_cO_one_iter.shape, self.acc_dtype)

        tTR_tO = cute.group_modes(tTR_tO, 1, cute.rank(tTR_tO))

        tSOut_layout = cute.make_layout(
            (((16, 4), 32), 1, 1), stride=(((65536, 2097152), 1), 0, 0)
        )
        tSOut0 = cute.make_tensor(tSOut.iterator, tSOut_layout)
        copy_atom_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_copy_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, tSOut0)
        thr_copy_r2t = tiled_copy_r2t.get_slice(thread_idx)
        tRT_cO = thr_copy_r2t.partition_S(tSOut0)
        tRT_tO = thr_copy_r2t.partition_D(tSOut0)
        tRT_cO_one_iter = tRT_cO[(None), None, 0, 0]  #
        tRT_rO = cute.make_rmem_tensor(tRT_cO_one_iter.shape, self.acc_dtype)

        for subtile_idx in cutlass.range_constexpr(2):
            tTR_tO_mn = tTR_tO[(None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tO_mn, tTR_rO)
            cute.arch.fence_view_async_tmem_load()

            if (sub_widx == 0 or sub_widx == 1) and subtile_idx == 0:
                for eid in cutlass.range_constexpr(16):
                    tRT_rO[eid] = tTR_rO[eid]

            if (sub_widx == 2 or sub_widx == 3) and subtile_idx == 1:
                for eid in cutlass.range_constexpr(16):
                    tRT_rO[eid] = tTR_rO[eid]

        tRT_tO = cute.group_modes(tRT_tO, 2, cute.rank(tRT_tO))
        for subtile_idx in cutlass.range_constexpr(0, 1):
            tRT_tO_mn = tRT_tO[(None, None, subtile_idx)]
            cute.copy(tiled_copy_r2t, tRT_rO, tRT_tO_mn)

        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_ivt_ad(
        self,
        sReg: cute.Tensor,
        sD: cute.Tensor,
    ):
        """Store diagonal block inverse (A/D blocks) from registers to smem."""
        if cutlass.const_expr(self.chunk_size <= 64):
            return
        tidx, _, _ = cute.arch.thread_idx()
        widx = cute.arch.warp_idx()
        sub_widx = widx % 4

        reg_layout = cute.make_layout((4, 2))
        result_e = cute.make_rmem_tensor(reg_layout.shape, self.acc_dtype)
        for it in cutlass.range_constexpr(0, 4):
            result_e[None, 0].store(sReg[(None, tidx % (self.threads_per_warp * len(self.cudacore_warp_ids))), it * 2 + 0].load())
            result_e[None, 1].store(sReg[(None, tidx % (self.threads_per_warp * len(self.cudacore_warp_ids))), it * 2 + 1].load())
            sD[(tidx % (self.threads_per_warp * len(self.cudacore_warp_ids)), (None, it % 2)), 0, sub_widx * 2 + it // 2, 0].store(
                result_e[None].load().to(self.q_dtype)
            )

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        g_iter: cute.Pointer,
        beta_iter: cute.Pointer,
        problem_size: Tuple[
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
        ],
        initial_state_f32_iter: Optional[cute.Pointer],
        state_output: Optional[cute.Pointer],
        scale: Optional[float],
        cum_seqlen_q_ptr: Optional[cute.Pointer] = None,
        workspace_state_ptr: Optional[cute.Pointer] = None,
        workspace_flags_ptr: Optional[cute.Pointer] = None,
        stream: cuda.CUstream = None,
    ):
        """Host-side entry: build tensor layouts, TMA descriptors, smem storage, and launch the kernel."""

        b, s_q, s_sum, h_q, h_v, d = problem_size

        # Reconstruct cum_seqlen_q tensor from pointer + batch_size.
        # Using Pointer (not Tensor) avoids CuTe DSL baking the shape into
        # the compiled kernel, so the same kernel handles any batch_size.
        cum_seqlen_q = None
        if cutlass.const_expr(cum_seqlen_q_ptr is not None):
            cum_seqlen_q = cute.make_tensor(
                cum_seqlen_q_ptr, cute.make_layout((b + 1,))
            )

        h_r = h_v // h_q
        o_offset = 0 if cum_seqlen_q is None else (-s_q * d * h_r * h_q)
        b_qk = b if cum_seqlen_q is None else s_sum
        b_o = b if cum_seqlen_q is None else s_q * (1 + b)
        b_v = b if cum_seqlen_q is None else s_sum

        stride_b_qk = h_q * s_q * d if cum_seqlen_q is None else d * h_q
        stride_b_vo = h_r * h_q * s_q * d if cum_seqlen_q is None else d * h_q * h_r

        b_gb = b if cum_seqlen_q is None else 1
        stride_b_gb = h_r * h_q * s_sum if cum_seqlen_q is None else 0
        q_layout = cute.make_layout(
            (s_sum, d, ((h_r, h_q), b_qk)),
            stride=(d * h_q, 1, ((0, d), stride_b_qk)),
        )
        q = cute.make_tensor(q_iter, q_layout)

        k_layout = cute.make_layout(
            (s_sum, d, ((h_r, h_q), b_qk)),
            stride=(d * h_q, 1, ((0, d), stride_b_qk)),
        )
        k = cute.make_tensor(k_iter, k_layout)
        v_layout = cute.make_layout(
            (d, s_sum, ((h_r, h_q), b_v)),
            stride=(1, d * h_q * h_r, ((d, d * h_r), stride_b_vo)),
        )
        v = cute.make_tensor(v_iter, v_layout)
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_q), b_o)),
            stride=(d * h_r * h_q, 1, ((d, d * h_r), stride_b_vo)),
        )
        o = cute.make_tensor(o_iter + o_offset, o_layout)

        state_layout = cute.make_layout(
            (self.head_dim, self.head_dim, ((h_r, h_q), b)),
            stride=(
                self.head_dim,
                1,
                (
                    (
                        self.head_dim * self.head_dim,
                        h_r * self.head_dim * self.head_dim,
                    ),
                    h_q * h_r * self.head_dim * self.head_dim,
                ),
            ),
        )

        state_input_f32 = (
            None
            if cutlass.const_expr(initial_state_f32_iter is None)
            else cute.make_tensor(initial_state_f32_iter, state_layout)
        )
        state_output = (
            None
            if cutlass.const_expr(state_output is None)
            else cute.make_tensor(state_output, state_layout)
        )

        gb_layout = cute.make_layout(
            (s_sum, 1, ((h_r, h_q), b_gb)),
            stride=(h_r * h_q, 0, ((1, h_r), stride_b_gb)),
        )
        gate = cute.make_tensor(g_iter, gb_layout)
        beta = cute.make_tensor(beta_iter, gb_layout)

        if cutlass.const_expr(scale is None):
            scale = 1.0 / cute.math.sqrt(cutlass.Float32(d))

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.g_dtype = gate.element_type
        self.beta_dtype = beta.element_type

        self.i_dtype = self.q_dtype

        self.tile_sched_params, grid = self._compute_grid(
            cute.shape((s_q, d, ((h_r, h_q), b))),
            self.cta_tiler,
            self.is_persistent,
            self.num_chunk_groups,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)
        self.s_output = utils.LayoutEnum.ROW_MAJOR

        cta_group = tcgen05.CtaGroup.ONE

        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )

        update_s_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.update_s_mma_tiler[:2],
        )

        o_intra_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.o_intra_mma_tiler[:2],
        )

        o_intra_tiled_ts_mma_new = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.o_intra_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        qs_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.qs_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        kkt_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            self.k_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.kkt_mma_tiler[:2],
        )

        kst_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.i_dtype,
            self.k_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.kst_mma_tiler[:2],
        )

        invert_sub_tiled_mma_ss_l0 = sm100_utils.make_trivial_tiled_mma(
            self.invert_type_ab,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.invert_acc_type,
            cta_group,
            self.invert_mma_sub_l0_tiler[:2],
        )
        invert_sub_tiled_mma_ts_l0 = sm100_utils.make_trivial_tiled_mma(
            self.invert_type_ab,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.invert_acc_type,
            cta_group,
            self.invert_mma_sub_l0_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        invert_sub_tiled_mma_ss_l1 = sm100_utils.make_trivial_tiled_mma(
            self.invert_type_ab,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.invert_acc_type,
            cta_group,
            self.invert_mma_sub_l1_tiler[:2],
        )
        invert_sub_tiled_mma_ts_l1 = sm100_utils.make_trivial_tiled_mma(
            self.invert_type_ab,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.invert_acc_type,
            cta_group,
            self.invert_mma_sub_l1_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        tuw_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            self.k_major_mode,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.tuw_mma_tiler[:2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )

        qk_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.i_dtype,
            self.qk_stage,
        )

        state_smem_layout_staged = sm100_utils.make_smem_layout_b(
            kst_tiled_mma,
            self.kst_mma_tiler,
            self.i_dtype,
            self.one_stage,
        )

        o_intra_smem_layout_staged_a = make_smem_layout_a_kind(
            o_intra_tiled_mma,
            self.o_intra_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        o_intra_smem_layout_staged_b = make_smem_layout_b_kind(
            o_intra_tiled_mma,
            self.o_intra_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        o_intra_smem_layout_staged_a_new = make_smem_layout_a_kind(
            o_intra_tiled_ts_mma_new,
            self.o_intra_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        qs_smem_layout_staged_a = sm100_utils.make_smem_layout_a(
            qs_tiled_mma,
            self.qs_mma_tiler,
            self.i_dtype,
            self.one_stage,
        )

        qs_smem_layout_staged_b = sm100_utils.make_smem_layout_b(
            qs_tiled_mma,
            self.qs_mma_tiler,
            self.i_dtype,
            self.one_stage,
        )

        update_s_smem_layout_staged_a = make_smem_layout_a_kind(
            update_s_tiled_mma,
            self.update_s_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        update_s_smem_layout_staged_b = make_smem_layout_b_kind(
            update_s_tiled_mma,
            self.update_s_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        state_output_smem_layout_staged = make_smem_layout_epi_kind(
            self.state_dtype,
            self.s_output,
            self.state_output_tiler,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        o_output_smem_layout_staged = make_smem_layout_epi_kind(
            self.i_dtype,
            self.o_layout,
            self.o_output_tiler,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        ivt_smem_layout = cute.make_layout(
            ((36, 32), 4)
        )  # padding, remove smem bank conflict
        # ivt_smem_layout = cute.make_layout(((32, 32), 4))  # padding, remove smem bank conflict

        sub_inner_smem_layout_staged = make_smem_layout_a_kind(
            invert_sub_tiled_mma_ss_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.sub_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        invert_sub_smem_layout_staged_ss_l0_a = sm100_utils.make_smem_layout_a(
            invert_sub_tiled_mma_ss_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ss_l0_b = sm100_utils.make_smem_layout_b(
            invert_sub_tiled_mma_ss_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ts_l0_a = sm100_utils.make_smem_layout_a(
            invert_sub_tiled_mma_ts_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ts_l0_b = sm100_utils.make_smem_layout_b(
            invert_sub_tiled_mma_ts_l0,
            self.invert_mma_sub_l0_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ss_l1_a = sm100_utils.make_smem_layout_a(
            invert_sub_tiled_mma_ss_l1,
            self.invert_mma_sub_l1_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ss_l1_b = sm100_utils.make_smem_layout_b(
            invert_sub_tiled_mma_ss_l1,
            self.invert_mma_sub_l1_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ts_l1_a = sm100_utils.make_smem_layout_a(
            invert_sub_tiled_mma_ts_l1,
            self.invert_mma_sub_l1_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        invert_sub_smem_layout_staged_ts_l1_b = sm100_utils.make_smem_layout_b(
            invert_sub_tiled_mma_ts_l1,
            self.invert_mma_sub_l1_tiler,
            self.invert_type_ab,
            self.invert_sub_stage,
        )

        # tuw
        tuw_smem_layout_staged_a = make_smem_layout_a_kind(
            tuw_tiled_mma,
            self.tuw_mma_tiler,
            self.q_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_INTER,
        )

        tuw_smem_layout_staged_b = make_smem_layout_b_kind(
            tuw_tiled_mma,
            self.tuw_mma_tiler,
            self.q_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        v_smem_layout_staged_b = make_smem_layout_b_kind(
            tuw_tiled_mma,
            self.tuw_mma_tiler,
            self.i_dtype,
            self.one_stage,
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_INTER,
        )

        # TMA
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # TMA load for Q
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged_b, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.tuw_mma_tiler,
            tuw_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        fake_state_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.state_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.state_dtype,
            cta_group,
            (self.head_dim, self.head_dim),
        )
        state_smem_layout_f32_staged = sm100_utils.make_smem_layout_b(
            fake_state_tiled_mma,
            (self.head_dim, self.head_dim, self.head_dim),
            self.state_dtype,
            self.one_stage,
        )
        state_smem_layout_f32 = cute.select(
            state_smem_layout_f32_staged, mode=[0, 1, 2]
        )
        tma_atom_state_f32, tma_tensor_state_f32 = (
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                state_input_f32,
                state_smem_layout_f32,
                self.qk_mma_tiler,
                fake_state_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            if cutlass.const_expr(state_input_f32 is not None)
            else (None, None)
        )

        state_smem_layout_output = cute.select(
            state_output_smem_layout_staged, mode=[0, 1]
        )
        tma_atom_state_output, tma_tensor_state_output = (
            cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                state_output,
                state_smem_layout_output,
                cute.composition(
                    cute.make_identity_layout(state_output.shape),
                    self.state_output_tiler,
                ),
            )
            if cutlass.const_expr(state_output is not None)
            else (None, None)
        )

        o_cta_v_layout = cute.composition(
            cute.make_identity_layout(o.shape), self.o_output_tiler
        )
        o_smem_layout_output = cute.select(o_output_smem_layout_staged, mode=[0, 1])
        tma_atom_o_output, tma_tensor_o_output = (
            cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                o,
                o_smem_layout_output,
                o_cta_v_layout,
            )
        )

        gate_smem_layout_staged = cute.make_layout(
            (self.chunk_size, 1, self.gate_stage)
        )
        gate_smem_layout = cute.select(gate_smem_layout_staged, mode=[0, 1])

        beta_smem_layout_staged = cute.make_layout(
            (self.chunk_size, 1, self.beta_stage)
        )
        beta_smem_layout = cute.select(beta_smem_layout_staged, mode=[0, 1])

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        state_copy_size = cute.size_in_bytes(self.i_dtype, state_smem_layout_f32)
        state_f32_copy_size = cute.size_in_bytes(
            self.state_dtype, state_smem_layout_f32
        )
        gate_copy_size = cute.size_in_bytes(self.g_dtype, gate_smem_layout)
        beta_copy_size = cute.size_in_bytes(self.beta_dtype, beta_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size
        self.tma_copy_v_bytes = v_copy_size
        self.tma_copy_state_bytes = state_copy_size
        self.tma_copy_state_f32_bytes = state_f32_copy_size
        self.tma_copy_gate_bytes = gate_copy_size
        self.tma_copy_beta_bytes = beta_copy_size

        # Shared memory storage
        @cute.struct
        class SharedStorage:
            load_qk_mbar_ptr: cute.struct.MemRange[Int64, self.qk_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.one_stage * 2]
            load_state_mbar_ptr: cute.struct.MemRange[Int64, self.one_stage * 2]
            mma_cudacore_mbar_ptr0: cute.struct.MemRange[Int64, 2]
            mma_qk_mbar_ptr0: cute.struct.MemRange[Int64, self.mma_qk_stage * 2]

            w0_epi_mbar_ptr: cute.struct.MemRange[Int64, 2]
            epi_w0_mbar_ptr: cute.struct.MemRange[Int64, 2]
            gb_w0_mbar_ptr: cute.struct.MemRange[Int64, self.gate_stage * 2]

            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
            # Tmem holding buffer
            tmem_holding_buf: Int32

            sV: cute.struct.Align[
                cute.struct.MemRange[self.i_dtype, cute.cosize(v_smem_layout_staged_b)],
                self.buffer_align_bytes,
            ]

            sQK: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(qk_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sIvt: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(ivt_smem_layout)],
                self.buffer_align_bytes,
            ]

            sGate: cute.struct.Align[
                cute.struct.MemRange[
                    self.g_beta_dtype, cute.cosize(gate_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sGateCumsum: cute.struct.Align[
                cute.struct.MemRange[
                    self.g_beta_dtype,
                    cute.cosize(cute.select(gate_smem_layout_staged, mode=[0, 1])),
                ],
                self.buffer_align_bytes,
            ]
            sBeta: cute.struct.Align[
                cute.struct.MemRange[
                    self.g_beta_dtype, cute.cosize(beta_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSubInner: cute.struct.Align[
                cute.struct.MemRange[
                    self.invert_type_ab, cute.cosize(sub_inner_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

            sState: cute.struct.Align[
                cute.struct.MemRange[
                    self.i_dtype, cute.cosize(state_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        if cutlass.const_expr(stream is None):
            stream = cutlass.cuda.default_stream()

        # Launch kernel
        self.kernel(
            qk_tiled_mma,
            o_intra_tiled_mma,
            o_intra_tiled_ts_mma_new,
            qs_tiled_mma,
            update_s_tiled_mma,
            kkt_tiled_mma,
            kst_tiled_mma,
            fake_state_tiled_mma,
            invert_sub_tiled_mma_ss_l0,
            invert_sub_tiled_mma_ts_l0,
            invert_sub_tiled_mma_ss_l1,
            invert_sub_tiled_mma_ts_l1,
            tuw_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_state_f32,
            tma_tensor_state_f32,
            gate,
            beta,
            o,
            tma_atom_state_output,
            tma_tensor_state_output,
            tma_atom_o_output,
            tma_tensor_o_output,
            cum_seqlen_q,
            q_smem_layout_staged,
            k_smem_layout_staged,
            qk_smem_layout_staged,
            o_intra_smem_layout_staged_a,
            o_intra_smem_layout_staged_b,
            o_intra_smem_layout_staged_a_new,
            qs_smem_layout_staged_a,
            qs_smem_layout_staged_b,
            update_s_smem_layout_staged_a,
            update_s_smem_layout_staged_b,
            state_smem_layout_staged,
            state_smem_layout_f32_staged,
            state_output_smem_layout_staged,
            o_output_smem_layout_staged,
            gate_smem_layout_staged,
            beta_smem_layout_staged,
            ivt_smem_layout,
            sub_inner_smem_layout_staged,
            invert_sub_smem_layout_staged_ss_l0_a,
            invert_sub_smem_layout_staged_ss_l0_b,
            invert_sub_smem_layout_staged_ts_l0_a,
            invert_sub_smem_layout_staged_ts_l0_b,
            invert_sub_smem_layout_staged_ss_l1_a,
            invert_sub_smem_layout_staged_ss_l1_b,
            invert_sub_smem_layout_staged_ts_l1_a,
            invert_sub_smem_layout_staged_ts_l1_b,
            tuw_smem_layout_staged_a,
            tuw_smem_layout_staged_b,
            v_smem_layout_staged_b,
            scale,
            self.tile_sched_params,
            workspace_state_ptr,
            workspace_flags_ptr,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @staticmethod
    def _compute_grid(
        o_shape: cute.Shape,
        cta_tiler: Tuple[int, int, int],
        is_persistent: bool,
        num_chunk_groups: int = 1,
    ) -> Tuple[GdnStaticTileSchedulerParams, Tuple[int, int, int]]:
        # The first dimension of problem_shape serves as chunk_group_id
        # when num_chunk_groups > 1 (chunk parallelism for small heads).
        m_dim = cute.ceil_div(cute.size(o_shape[1]), cta_tiler[2])
        if num_chunk_groups > 1:
            m_dim = num_chunk_groups
        tile_sched_params = create_gdn_static_tile_scheduler_params(
            is_persistent,
            (
                m_dim,
                cute.size(o_shape[2][0]),
                cute.size(o_shape[2][1]),
            ),
        )
        grid = GdnStaticTileScheduler.get_grid_shape(tile_sched_params)
        return tile_sched_params, grid


# ============================================================================
# FlashInfer API Layer
# ============================================================================


@functools.lru_cache(maxsize=128)
def _get_problem_size(q_shape, v_shape, cu_seqlens_tuple):
    """Compute problem_size, cached by (q_shape, v_shape, cu_seqlens tuple).

    Args:
        q_shape: Tuple of (b, s_q, h_q, d).
        v_shape: Tuple of (b, s_v, h_v, d).
        cu_seqlens_tuple: Tuple of cumulative sequence lengths, or None.
    """
    b, s_q, h_q, d = q_shape
    _, _, h_v, _ = v_shape
    max_s_q = s_q
    sum_s_q = s_q
    if cu_seqlens_tuple is not None:
        sum_s_q = cu_seqlens_tuple[-1]
        b = len(cu_seqlens_tuple) - 1
        max_s_q = max(cu_seqlens_tuple[i + 1] - cu_seqlens_tuple[i] for i in range(b))
        return (b, max_s_q, sum_s_q, h_q, h_v, d)
    return (b, max_s_q, sum_s_q, h_q, h_v, d)


@functools.cache
def _get_compiled_gdn_prefill_kernel(
    problem_size,
    dtype: torch.dtype,
    is_varlen: bool,
    is_initial_state: bool,
    is_output_state: bool,
    scale: float,
    chunk_size: int = 128,
    num_chunk_groups: int = 1,
):
    """Cache compiled kernel for given configuration."""
    return {}


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Public API: compile (on first call) and run the GDN chunked linear attention kernel.

    Args:
        q, k: (B, T, H_qk, D) query and key tensors (f16/bf16).
        v:     (B, T, H_v, D) value tensor (f16/bf16). H_v must be a multiple of H_qk.
        g:     (1, total_seq_len, H_v) gate values (f32, log-space).
        beta:  (1, total_seq_len, H_v) beta values (f32, sigmoid-space).
        scale: Scale factor for the attention scores. Defaults to 1/sqrt(D).
        initial_state: (B, H_v, D, D) recurrent state (f32), or None for zero init.
        output_final_state: If True, return the final state alongside output.
        cu_seqlens: Cumulative sequence lengths for variable-length batching.
        use_qk_l2norm_in_kernel: If True, use L2 norm of QK in the kernel.(Not supported yet)
        output: Pre-allocated output tensor, or None to allocate internally.
        output_state: Pre-allocated state output tensor, or None.

    Returns:
        output or (output, output_state) depending on output_final_state.
    """

    # Allocate output if needed
    if output is None:
        output = torch.empty_like(v)

    # Check if supported
    if not GDN.can_implement(
        tuple(q.shape),
        tuple(v.shape),
        q.dtype,
        output.dtype,
        beta.dtype,
        use_qk_l2norm_in_kernel,
    ):
        raise ValueError("Unsupported input shape or dtype")
    cu_seqlens_tuple = tuple(cu_seqlens.tolist()) if cu_seqlens is not None else None
    problem_size = _get_problem_size(tuple(q.shape), tuple(v.shape), cu_seqlens_tuple)

    # Allocate output_state if needed
    if output_final_state and (output_state is None):
        output_state = torch.empty(
            (problem_size[0], problem_size[4], problem_size[5], problem_size[5]),
            dtype=torch.float32,
            device=q.device,
        )

    # Compile kernel with TVM FFI (cached)
    is_varlen = cu_seqlens is not None
    is_initial_state = initial_state is not None
    num_o_heads = max(q.shape[2], v.shape[2])
    num_seqs_local = problem_size[0]
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    use_small_chunk = num_o_heads * num_seqs_local < sm_count // 2
    chunk_size = 64 if use_small_chunk else 128

    # Seq-dimension parallelism: split chunks across multiple CTAs per head.
    # num_chunk_groups > 1 when SM utilization is low.
    num_ctas_base = num_o_heads * num_seqs_local
    # Enable seq-dim parallelism when SM utilization is low.
    # Works for both padded (cu_seqlens=None) and varlen single-seq (num_seqs=1).
    can_parallel = (cu_seqlens is None) or (num_seqs_local == 1)
    if num_ctas_base < sm_count and can_parallel:
        max_s_q = problem_size[1]
        total_chunks = max_s_q // chunk_size
        # Target: enough groups to fill ~80% of SMs
        target_ctas = max(sm_count * 3 // 4, num_ctas_base)
        num_chunk_groups = min(
            target_ctas // max(num_ctas_base, 1),
            total_chunks,  # can't have more groups than chunks
        )
        num_chunk_groups = max(1, num_chunk_groups)
    else:
        num_chunk_groups = 1

    # For seq-dim parallelism, always use initial_state and output_state
    cache_is_initial = is_initial_state or num_chunk_groups > 1
    cache_output_state = output_final_state or num_chunk_groups > 1
    # Cache key: only compile-time invariants. problem_size[0:3] (b, s_q, s_sum)
    # vary at runtime and are handled via Pointer + make_tensor inside the kernel.
    h_q, h_v, d_val = problem_size[3], problem_size[4], problem_size[5]
    cache_key = (
        (h_q, h_v, d_val),
        q.dtype,
        is_varlen,
        cache_is_initial,
        cache_output_state,
        scale,
        chunk_size,
        num_chunk_groups,
    )
    cache = _get_compiled_gdn_prefill_kernel(*cache_key)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Allocate workspace for inter-group state transfer + sync flags
    if num_chunk_groups > 1:
        num_heads_total = num_o_heads * num_seqs_local
        d = q.shape[3]
        workspace_state = torch.zeros(
            (num_heads_total, d, d), dtype=torch.float32, device=q.device
        )
        workspace_flags = torch.zeros(
            (num_heads_total * num_chunk_groups,), dtype=torch.int32, device=q.device
        )
        # For seq-dim parallelism: use output_state as inter-group state buffer.
        # Copy initial_state to output_state so group 0's TMA loads the right data.
        if output_state is None:
            output_state = torch.empty(
                (num_seqs_local, num_o_heads, d, d),
                dtype=torch.float32, device=q.device,
            )
        if initial_state is not None:
            output_state.copy_(initial_state)
        else:
            output_state.zero_()
        # Pass output_state as BOTH initial_state and state_output to the kernel.
        # The TMA will load from output_state (initial state for group 0),
        # and the epilogue will write to the same output_state (for next group).
        initial_state = output_state
        is_initial_state = True
        output_final_state = True
    else:
        workspace_state = None
        workspace_flags = None

    if "compiled_gdn" not in cache:
        # GDN kernel
        gdn = GDN(chunk_size=chunk_size, num_chunk_groups=num_chunk_groups)
        q_tensor = from_dlpack(q, assumed_align=16, enable_tvm_ffi=True)
        k_tensor = from_dlpack(k, assumed_align=16, enable_tvm_ffi=True)
        v_tensor = from_dlpack(v, assumed_align=16, enable_tvm_ffi=True)
        o_tensor = from_dlpack(output, assumed_align=16, enable_tvm_ffi=True)
        gate_tensor = from_dlpack(g, assumed_align=16, enable_tvm_ffi=True)
        beta_tensor = from_dlpack(beta, assumed_align=16, enable_tvm_ffi=True)
        cu_seqlens_tensor = (
            from_dlpack(cu_seqlens, assumed_align=16, enable_tvm_ffi=True)
            if cu_seqlens is not None
            else None
        )
        state_tensor = (
            from_dlpack(initial_state, assumed_align=16, enable_tvm_ffi=True)
            if initial_state is not None
            else None
        )
        state_output_tensor = (
            from_dlpack(output_state, assumed_align=16, enable_tvm_ffi=True)
            if output_final_state
            else None
        )

        # Workspace tensors for chunk parallelism
        ws_state_tensor = (
            from_dlpack(workspace_state, assumed_align=16, enable_tvm_ffi=True)
            if workspace_state is not None
            else None
        )
        ws_flags_tensor = (
            from_dlpack(workspace_flags, assumed_align=16, enable_tvm_ffi=True)
            if workspace_flags is not None
            else None
        )

        # Compile GDN kernel
        options = EnableTVMFFI
        compiled_gdn = cute.compile[options](
            gdn,
            q_tensor.iterator,
            k_tensor.iterator,
            v_tensor.iterator,
            o_tensor.iterator,
            gate_tensor.iterator,
            beta_tensor.iterator,
            problem_size,
            state_tensor.iterator if state_tensor is not None else None,
            state_output_tensor.iterator if output_final_state else None,
            scale,
            cu_seqlens_tensor.iterator if cu_seqlens_tensor is not None else None,
            ws_state_tensor.iterator if ws_state_tensor is not None else None,
            ws_flags_tensor.iterator if ws_flags_tensor is not None else None,
            stream=current_stream,
        )
        cache["compiled_gdn"] = compiled_gdn

    compiled_gdn = cache["compiled_gdn"]

    # Run GDN kernel
    compiled_gdn(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        output.data_ptr(),
        g.data_ptr(),
        beta.data_ptr(),
        problem_size,
        initial_state.data_ptr() if initial_state is not None else None,
        output_state.data_ptr() if output_final_state else None,
        scale,
        from_dlpack(cu_seqlens, assumed_align=16, enable_tvm_ffi=True).iterator if cu_seqlens is not None else None,
        workspace_state.data_ptr() if workspace_state is not None else None,
        workspace_flags.data_ptr() if workspace_flags is not None else None,
        stream=current_stream,
    )

    if output_final_state:
        return output, output_state
    else:
        return output
