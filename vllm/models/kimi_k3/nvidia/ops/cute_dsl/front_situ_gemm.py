# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Derived from cudnn-frontend 1.32.3
# `cudnn/gemm/cutedsl/dense/swiglu/dense_gemm_persistent_swiglu.py`:
#   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
# Modified by NVIDIA: gate/up pairing read from the stock [gate; up] weight layout,
# the Kimi SiTU-and-mul epilogue with the stock kernel's FP32 operation order, an
# optional second weight (latent down projection) in the same persistent launch, a
# dynamic token count, and no raw GEMM output for the gate/up tiles.
"""Kimi K3 MoE front GEMM: shared gate_up + SiTU-and-mul (+ latent down), SM100.

One persistent 2-CTA tcgen05 BF16 GEMM over the token activations x [M, K]:

* shared-expert tiles: the 2-CTA MMA splits B by N across the CTA pair, so with a
  BN-wide tile CTA 0 loads BN/2 gate rows and CTA 1 the matching BN/2 up rows of
  the stock [gate; up] weight (no weight permutation). The accumulator then holds
  gate and up for the same BN/2 intermediate columns. The epilogue rounds both to
  BF16 (what the stock GEMM writes) and applies the stock SiTU-and-mul in FP32 with
  the same operation sequence as `situ_and_mul_kernel`, writing act [M, I].
* latent tiles (optional): the plain BF16 GEMM output of the latent down
  projection, latent [M, LAT].

Per element this is the stock computation: FP32 accumulation over K in order (no
split-K), BF16 rounding, then the same FP32 activation. Outputs are compared
bitwise against cuBLAS + `situ_and_mul` in the tests; token counts where cuBLAS
itself switches to a different K reduction should keep the stock path.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import numpy as np
from cutlass import BFloat16, Float32
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05

try:
    from cutlass.tensor_utils import LayoutEnum
except ModuleNotFoundError:  # older cutlass-dsl
    from cutlass.utils import LayoutEnum
try:
    from cutlass.memory import (
        SmemAllocator,
        get_num_tmem_alloc_cols,
        get_smem_capacity_in_bytes,
    )
except ModuleNotFoundError:  # older cutlass-dsl
    from cutlass.utils import (
        SmemAllocator,
        get_num_tmem_alloc_cols,
        get_smem_capacity_in_bytes,
    )


class SituConstants:
    """FP32 constants of `situ_and_mul(out, x, beta, linear_beta)`."""

    def __init__(self, beta: float, linear_beta: float | None):
        b = np.float32(beta)
        self.half_beta = float(np.float32(0.5) * b)
        self.inv_beta = float(np.float32(1.0) / b)
        self.clamp_up = linear_beta is not None and linear_beta > 0
        lb = np.float32(linear_beta if self.clamp_up else 0.0)
        self.linear_beta = float(lb)
        self.inv_linear_beta = float(np.float32(1.0) / lb) if self.clamp_up else 0.0


def situ_f32(g, u, c: SituConstants):
    """`situ_activation` of csrc/activation_kernels.cu in the same FP32 order
    (accurate tanh, as the stock kernel is built without fast math)."""
    tanh = cute.math.tanh
    gate = (Float32(c.half_beta) * tanh(g * Float32(c.inv_beta))) * (
        Float32(1.0) + tanh(g * Float32(0.5))
    )
    if c.clamp_up:
        up = Float32(c.linear_beta) * tanh(u * Float32(c.inv_linear_beta))
    else:
        up = u
    return gate * up


class KimiFrontSituGemm:
    """x [M, K] @ [gate; up]^T -> SiTU-and-mul act [M, I]; optionally also
    x @ W_lat^T -> latent [M, LAT] in the same launch.

    2-CTA tcgen05 MMA with a (256, BN) tile; BN/2 must divide I, BN must be a
    multiple of 64 (two epilogue subtiles per half).
    """

    def __init__(
        self,
        n_tile: int = 192,
        with_latent: bool = True,
        beta: float = 4.0,
        linear_beta: float | None = 25.0,
        num_acc_stage: int = 2,
        num_c_stage: int = 4,
    ):
        assert n_tile % 64 == 0 and 64 <= n_tile <= 256
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cluster_shape_mn = (2, 1)
        self.mma_tiler = (256, n_tile, 1)
        self.half_n = n_tile // 2
        self.with_latent = with_latent
        self.situ = SituConstants(beta, linear_beta)
        self.num_acc_stage = num_acc_stage
        self.num_c_stage = num_c_stage
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * 6
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = get_smem_capacity_in_bytes("sm_100")

    def _setup_attributes(self):
        self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        tiled_mma = self.tiled_mma
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * 4)
        self.cta_tile_shape_c = (self.mma_tiler[0] // 2, self.half_n, self.mma_tiler[2])
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        # One epilogue subtile shape for both outputs: (128, EPI_N) with EPI_N | BN/2.
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_c, True, LayoutEnum.ROW_MAJOR, BFloat16
        )
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, BFloat16, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, BFloat16, 1)
        c_one = sm100_utils.make_smem_layout_epi(
            BFloat16, LayoutEnum.ROW_MAJOR, self.epi_tile, 1
        )
        ab_bytes = cute.size_in_bytes(BFloat16, a_one) + cute.size_in_bytes(
            BFloat16, b_one
        )
        c_bytes = cute.size_in_bytes(BFloat16, c_one) * self.num_c_stage
        self.num_ab_stage = (self.smem_capacity - 1024 - c_bytes) // ab_bytes
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, BFloat16, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, BFloat16, self.num_ab_stage
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            BFloat16, LayoutEnum.ROW_MAJOR, self.epi_tile, self.num_c_stage
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        self.num_tmem_alloc_cols = get_num_tmem_alloc_cols(
            tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,  # [M, K] BF16, K-major
        b_gu: cute.Tensor,  # [2I, K] BF16 stock [gate; up]
        b_lat: cute.Tensor,  # [LAT, K] BF16 (ignored without latent)
        c_act: cute.Tensor,  # [M, I] BF16
        c_lat: cute.Tensor,  # [M, LAT] BF16 (ignored without latent)
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        tiled_mma = self.tiled_mma
        M = a.shape[0]
        K = a.shape[1]
        two_i = b_gu.shape[0]
        inter = two_i // 2
        half_n = self.half_n
        n_gu = inter // half_n
        n_lat = 0
        if cutlass.const_expr(self.with_latent):
            n_lat = cute.ceil_div(b_lat.shape[0], self.mma_tiler[1])

        # (M, K, L) views
        mA = cute.make_tensor(
            a.iterator, cute.make_layout((M, K, 1), stride=(a.stride[0], 1, 0))
        )
        # B for the shared-expert tiles: row n' = j*BN + h*BN/2 + r of the
        # virtual operand is weight row h*I + j*BN/2 + r (h = 0 gate, 1 up).
        sb = b_gu.stride[0]
        mB_gu = cute.make_tensor(
            b_gu.iterator,
            cute.make_layout(
                ((half_n, 2, n_gu), K, 1),
                stride=((sb, inter * sb, half_n * sb), 1, 0),
            ),
        )
        mC_act = cute.make_tensor(
            c_act.iterator,
            cute.make_layout((M, inter, 1), stride=(c_act.stride[0], 1, 0)),
        )
        mB_lat = None
        mC_lat = None
        if cutlass.const_expr(self.with_latent):
            mB_lat = cute.make_tensor(
                b_lat.iterator,
                cute.make_layout(
                    (b_lat.shape[0], K, 1), stride=(b_lat.stride[0], 1, 0)
                ),
            )
            mC_lat = cute.make_tensor(
                c_lat.iterator,
                cute.make_layout(
                    (M, b_lat.shape[0], 1), stride=(c_lat.stride[0], 1, 0)
                ),
            )

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b_gu, tma_tensor_b_gu = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB_gu,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b_lat = None
        tma_tensor_b_lat = None
        if cutlass.const_expr(self.with_latent):
            tma_atom_b_lat, tma_tensor_b_lat = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                mB_lat,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        self.num_tma_load_bytes = (
            cute.size_in_bytes(BFloat16, a_smem_layout)
            + cute.size_in_bytes(BFloat16, b_smem_layout)
        ) * 2

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_act, tma_tensor_act = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mC_act,
            epi_smem_layout,
            cute.composition(cute.make_identity_layout(mC_act.shape), self.epi_tile),
        )
        tma_atom_lat = None
        tma_tensor_lat = None
        if cutlass.const_expr(self.with_latent):
            tma_atom_lat, tma_tensor_lat = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mC_lat,
                epi_smem_layout,
                cute.composition(
                    cute.make_identity_layout(mC_lat.shape), self.epi_tile
                ),
            )

        num_ctas_mnl = (cute.ceil_div(M, self.mma_tiler[0]) * 2, n_gu + n_lat, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, (*self.cluster_shape_mn, 1)
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    BFloat16, cute.cosize(self.c_smem_layout_staged.outer)
                ],
                1024,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    BFloat16, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    BFloat16, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                1024,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b_gu,
            tma_tensor_b_gu,
            tma_atom_b_lat,
            tma_tensor_b_lat,
            tma_atom_act,
            tma_tensor_act,
            tma_atom_lat,
            tma_tensor_lat,
            n_gu,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b_gu: cute.CopyAtom,
        mB_gu: cute.Tensor,
        tma_atom_b_lat,
        mB_lat,
        tma_atom_act: cute.CopyAtom,
        mAct: cute.Tensor,
        tma_atom_lat,
        mLat,
        n_gu: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b_gu)
            cpasync.prefetch_descriptor(tma_atom_act)
            if cutlass.const_expr(self.with_latent):
                cpasync.prefetch_descriptor(tma_atom_b_lat)
                cpasync.prefetch_descriptor(tma_atom_lat)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % 2
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.ptr
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_producer
            ),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id) * 2
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        if warp_idx == self.tma_warp_id:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_gu = cute.local_tile(
            mB_gu, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gAct = cute.local_tile(
            mAct, (self.mma_tiler[0], self.half_n), (None, None, None)
        )
        k_block_cnt = cute.size(gA_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB_gu = thr_mma.partition_B(gB_gu)
        tCgAct = thr_mma.partition_C(gAct)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB_gu = cpasync.tma_partition(
            tma_atom_b_gu,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB_gu, 0, 3),
        )
        tBgB_lat = None
        tCgLat = None
        if cutlass.const_expr(self.with_latent):
            gB_lat = cute.local_tile(
                mB_lat, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            tCgB_lat = thr_mma.partition_B(gB_lat)
            _, tBgB_lat = cpasync.tma_partition(
                tma_atom_b_lat,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB_lat, 0, 3),
            )
            gLat = cute.local_tile(
                mLat, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
            )
            tCgLat = thr_mma.partition_C(gLat)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        cute.arch.cluster_wait()

        # ---------------- TMA load warp ----------------
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            while work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                m_tile = cur[0] // 2
                n_tile = cur[1]
                tAgA_slice = tAgA[(None, m_tile, None, cur[2])]
                ab_producer_state.reset_count()
                peek = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek = ab_pipeline.producer_try_acquire(ab_producer_state)
                for _k in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek)
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    if n_tile < n_gu:
                        cute.copy(
                            tma_atom_b_gu,
                            tBgB_gu[(None, n_tile, ab_producer_state.count, cur[2])],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=b_full_mcast_mask,
                        )
                    else:
                        if cutlass.const_expr(self.with_latent):
                            cute.copy(
                                tma_atom_b_lat,
                                tBgB_lat[
                                    (
                                        None,
                                        n_tile - n_gu,
                                        ab_producer_state.count,
                                        cur[2],
                                    )
                                ],
                                tBsB[(None, ab_producer_state.index)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                            )
                    ab_producer_state.advance()
                    peek = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek = ab_pipeline.producer_try_acquire(ab_producer_state)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_pipeline.producer_tail(ab_producer_state)

        # ---------------- MMA warp ----------------
        if warp_idx == self.mma_warp_id:
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id, number_of_threads=32 * 5
            )
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                ab_consumer_state.reset_count()
                peek = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for _k in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek)
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kp in cutlass.range(num_kphases, unroll_full=True):
                            kc = (None, None, kp, ab_consumer_state.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB[kc], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    peek = cutlass.Boolean(1)
                    # Kept nested as in the kernel this file is derived from.
                    if ab_consumer_state.count < k_block_cnt:  # noqa: SIM102
                        if is_leader_cta:
                            peek = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        # ---------------- epilogue warps ----------------
        if warp_idx < self.mma_warp_id:
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=True
                )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id, number_of_threads=32 * 5
            )
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # TMEM -> RMEM
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_c,
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                self.acc_dtype,
                epi_tile,
                True,
            )
            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], epi_tile
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            gAct_epi = cute.flat_divide(
                tCgAct[((None, None), 0, 0, None, None, None)], epi_tile
            )
            tTR_gAct = thr_copy_t2r.partition_D(gAct_epi)
            frag_shape = tTR_gAct[(None, None, None, 0, 0, 0, 0, 0)].shape
            tTR_rG = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            tTR_rU = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            tTR_rOut = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            tTR_rC = cute.make_rmem_tensor(frag_shape, BFloat16)

            # RMEM -> SMEM
            copy_atom_r2s = sm100_utils.get_smem_store_op(
                LayoutEnum.ROW_MAJOR, BFloat16, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            # SMEM -> GMEM (TMA)
            bSG_sC, bSG_gAct_all = cpasync.tma_partition(
                tma_atom_act,
                0,
                cute.make_layout(1),
                cute.group_modes(sC, 0, 2),
                cute.group_modes(gAct_epi, 0, 2),
            )
            bSG_gLat_all = None
            if cutlass.const_expr(self.with_latent):
                gLat_epi = cute.flat_divide(
                    tCgLat[((None, None), 0, 0, None, None, None)], epi_tile
                )
                _, bSG_gLat_all = cpasync.tma_partition(
                    tma_atom_lat,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    cute.group_modes(gLat_epi, 0, 2),
                )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epilog_warp_id)
                ),
            )
            epi_threads = 32 * len(self.epilog_warp_id)
            sub_half = cute.size(tTR_tAcc_base.shape, mode=[4]) // 2
            store_idx = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                m_tile = cur[0] // 2
                n_tile = cur[1]
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                if n_tile < n_gu:
                    bSG_gAct = bSG_gAct_all[(None, None, None, m_tile, n_tile, cur[2])]
                    bSG_gAct = cute.group_modes(bSG_gAct, 1, cute.rank(bSG_gAct))
                    for s in cutlass.range(sub_half, unroll=1):
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rG
                        )
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, s + sub_half)],
                            tTR_rU,
                        )
                        # The stock GEMM's BF16 output, back in FP32.
                        tTR_rG.store(tTR_rG.load().to(BFloat16).to(Float32))
                        tTR_rU.store(tTR_rU.load().to(BFloat16).to(Float32))
                        for i in cutlass.range_constexpr(cute.size(tTR_rOut)):
                            tTR_rOut[i] = situ_f32(tTR_rG[i], tTR_rU[i], self.situ)
                        tRS_rC.store(
                            tiled_copy_r2s.retile(tTR_rOut).load().to(BFloat16)
                        )
                        buf = store_idx % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, buf)]
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epi_threads,
                        )
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_act, bSG_sC[(None, buf)], bSG_gAct[(None, s)]
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epi_threads,
                        )
                        store_idx = store_idx + 1
                else:
                    if cutlass.const_expr(self.with_latent):
                        bSG_gLat = bSG_gLat_all[
                            (None, None, None, m_tile, n_tile - n_gu, cur[2])
                        ]
                        bSG_gLat = cute.group_modes(bSG_gLat, 1, cute.rank(bSG_gLat))
                        for s in cutlass.range(2 * sub_half, unroll=1):
                            cute.copy(
                                tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rG
                            )
                            tRS_rC.store(
                                tiled_copy_r2s.retile(tTR_rG).load().to(BFloat16)
                            )
                            buf = store_idx % self.num_c_stage
                            cute.copy(
                                tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, buf)]
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cute.arch.barrier(
                                barrier_id=self.epilog_sync_bar_id,
                                number_of_threads=epi_threads,
                            )
                            if warp_idx == self.epilog_warp_id[0]:
                                cute.copy(
                                    tma_atom_lat,
                                    bSG_sC[(None, buf)],
                                    bSG_gLat[(None, s)],
                                )
                                c_pipeline.producer_commit()
                                c_pipeline.producer_acquire()
                            cute.arch.barrier(
                                barrier_id=self.epilog_sync_bar_id,
                                number_of_threads=epi_threads,
                            )
                            store_idx = store_idx + 1
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.mbarrier_arrive(
                    tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                )
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=True
                )
            c_pipeline.producer_tail()


_COMPILED: dict = {}


def compile_front_situ_gemm(
    inter: int,
    hidden: int,
    latent: int | None,
    *,
    n_tile: int = 192,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
    num_c_stage: int = 4,
):
    """Compile once for a dynamic token count M (fixed weight shapes)."""
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    key = (inter, hidden, latent, n_tile, beta, linear_beta, num_c_stage)
    fn = _COMPILED.get(key)
    if fn is not None:
        return fn
    M = cute.sym_int()
    a = make_fake_tensor(
        BFloat16, (M, hidden), (cute.sym_int64(divisibility=8), 1), assumed_align=16
    )
    b_gu = make_fake_tensor(
        BFloat16, (2 * inter, hidden), (hidden, 1), assumed_align=16
    )
    act = make_fake_tensor(
        BFloat16, (M, inter), (cute.sym_int64(divisibility=8), 1), assumed_align=16
    )
    if latent is not None:
        b_lat = make_fake_tensor(
            BFloat16, (latent, hidden), (hidden, 1), assumed_align=16
        )
        lat = make_fake_tensor(
            BFloat16, (M, latent), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
    else:
        b_lat = make_fake_tensor(BFloat16, (64, hidden), (hidden, 1), assumed_align=16)
        lat = make_fake_tensor(
            BFloat16, (M, 64), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
    kernel = KimiFrontSituGemm(
        n_tile=n_tile,
        with_latent=latent is not None,
        beta=beta,
        linear_beta=linear_beta,
        num_c_stage=num_c_stage,
    )
    max_clusters = utils.HardwareInfo().get_max_active_clusters(2)
    fn = cute.compile(
        kernel,
        a,
        b_gu,
        b_lat,
        act,
        lat,
        max_clusters,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _COMPILED[key] = fn
    return fn
