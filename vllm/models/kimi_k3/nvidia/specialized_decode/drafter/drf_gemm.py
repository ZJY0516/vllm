# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-layer GEMV with fused prologue/epilogue work (K31_DRF).

The decode GEMV of `cutedsl/gemm/warp_splitk.py` (cp.async weight ring, four
mma.sync warps holding split-K partials, one BF16 rounding), with options that
remove the small kernels around the draft layers' linears:

* `norm` (RMSNorm into the consumer): the activation is the residual stream `r`
  (the all-reduce output), the weight carries the RMSNorm weight folded per
  column (`W * gamma`, one BF16 rounding), and every program sums r^2 of its
  activation rows from the shared-memory stages it already holds; the epilogue
  multiplies the FP32 accumulator by rsqrt(mean(r^2) + eps) (vLLM rms_norm's
  rsqrt) before the BF16 rounding.
* `resid` (residual add before the all-reduce): the epilogue adds a BF16 row
  `residual[token, feature]` to the FP32 accumulator (one rank of the TP group
  passes it, so the all-reduce returns `x + residual`).
* `act` (SiLU-and-mul): the weight rows are permuted so each output tile holds
  T/2 gate rows then the T/2 matching up rows; the epilogue rounds gate and up to
  BF16 (as the gate_up output), takes silu(gate) from a 65,536-entry table made
  by vLLM's own `silu_and_mul` (exact), multiplies by `up + 0` in FP32 and rounds
  once (vLLM act_and_mul's arithmetic), writing N/2 columns.

`act` alone is bit-identical to GEMV + act_and_mul; `norm` and `resid` change
where the roundings happen (not bit-identical).
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch as _torch
from cutlass import Float32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    bits_float,
    gate_bf16_bits,
    rms_norm_rstd,
    table_load,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.warp_splitk import (
    _A_LOADER_THREADS,
    _A_LOADER_WARPS,
    _COMPUTE_WARPS,
    _MAILBOX_GROUP_VALUES,
    _MAILBOX_PADDED_GROUP_VALUES,
    _MAX_M,
    _MMA_SHAPE,
    WarpSplitKTactic,
    _a_fragment_offset,
    _b_pair_offset,
    _issue_weight_tile,
    _l2_policy_evict_first,
    _ldmatrix_x4,
    _make_smem_layout,
    _partial_offset,
    _smem_bytes,
    check_tactic,
)


def _drf_smem_bytes(tactic: WarpSplitKTactic) -> int:
    return _smem_bytes(tactic) + _COMPUTE_WARPS * tactic.token_tile * 4 + 16


class DrfWarpSplitKKernel:
    def __init__(self, tactic: WarpSplitKTactic, *, norm: bool, resid: bool, act: bool,
                 eps: float) -> None:
        self.output_tile = tactic.output_tile
        self.token_tile = tactic.token_tile
        self.k_tile = tactic.k_tile
        self.stages = tactic.stages
        self.b_loader_warps = tactic.b_loader_warps
        self.compute_warp_base = _A_LOADER_WARPS + self.b_loader_warps
        self.threads = (self.compute_warp_base + _COMPUTE_WARPS) * 32
        self.load_threads = self.compute_warp_base * 32
        self.norm, self.resid, self.act = norm, resid, act
        self.eps = float(eps)
        self.smem_bytes = _drf_smem_bytes(tactic)

    @cute.experimental.jit
    def __call__(self, weight, activation, output, residual, table,
                 stream: _cuda.CUstream):
        sA_layout = _make_smem_layout(
            weight.element_type, self.output_tile, self.k_tile, self.stages
        )
        sB_layout = _make_smem_layout(
            activation.element_type, self.token_tile, self.k_tile, self.stages
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(weight.element_type, cutlass.Float32, _MMA_SHAPE),
            cute.make_layout((1, 1, _COMPUTE_WARPS)),
            permutation_mnk=(self.output_tile, self.token_tile, self.k_tile),
        )
        self.kernel(
            weight, activation, output, residual, table, sA_layout, sB_layout,
            tiled_mma,
        ).launch(
            grid=(
                cute.ceil_div(weight.shape[0], self.output_tile),
                cute.ceil_div(activation.shape[0], self.token_tile),
                1,
            ),
            block=(self.threads, 1, 1),
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        weight: cute.Tensor,
        activation: cute.Tensor,
        output: cute.Tensor,
        residual: cute.Tensor,
        table: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        output_tile_idx, token_tile_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE
        reuse_stages = cute.size(weight, mode=[1]) > self.stages * self.k_tile

        sA = cute_ext.allocate(
            weight.element_type, cute.AddressSpace.smem, sA_layout, alignment=1024
        )
        sB = cute_ext.allocate(
            activation.element_type, cute.AddressSpace.smem, sB_layout, alignment=1024
        )
        values_per_lane = self.output_tile * self.token_tile // cute.arch.WARP_SIZE
        partial_stride = (
            self.output_tile * self.token_tile // _MAILBOX_GROUP_VALUES
        ) * _MAILBOX_PADDED_GROUP_VALUES
        partials = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(_COMPUTE_WARPS * partial_stride),
            alignment=16,
        )
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(self.stages),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(self.stages),
            alignment=8,
        )
        sq = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.smem,
            cute.make_layout(_COMPUTE_WARPS * self.token_tile), alignment=16,
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator

        k_tile_count = cute.size(weight, mode=[1]) // self.k_tile

        if warp_idx < _A_LOADER_WARPS:
            _issue_weight_tile(
                sA, weight, tidx, output_tile_idx, 0, 0, cutlass.Int64(0),
                self.output_tile, self.k_tile, False,
            )
        if warp_idx == self.compute_warp_base:  # noqa: SIM102
            if lane_idx < self.stages:
                cute.arch.mbarrier_init(bar_full + lane_idx, self.load_threads)
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_init(bar_empty + lane_idx, _COMPUTE_WARPS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx < _A_LOADER_WARPS:
            weight_policy = _l2_policy_evict_first()
            empty_phase = cutlass.Int32(1)
            prims.cp_async_mbarrier_arrive(bar_full.llvm_ptr, noinc=True)
            for k_tile in cutlass.range(1, k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                _issue_weight_tile(
                    sA, weight, tidx, output_tile_idx, k_tile, stage, weight_policy,
                    self.output_tile, self.k_tile, True,
                )
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
        elif warp_idx < self.compute_warp_base:
            cute.arch.griddepcontrol_wait()
            b_loader_threads = self.b_loader_warps * 32
            local_tid = tidx - _A_LOADER_WARPS * 32
            copy_count = self.token_tile * self.k_tile // (8 * b_loader_threads)
            empty_phase = cutlass.Int32(1)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                for copy_idx in cutlass.range_constexpr(copy_count):
                    linear = local_tid * 8 + copy_idx * b_loader_threads * 8
                    row = linear // self.k_tile
                    col = linear % self.k_tile
                    global_row = token_tile_idx * self.token_tile + row
                    if global_row < cute.size(activation, mode=[0]):
                        dst = (
                            sB.iterator.raw_ptr()
                            + stage * self.token_tile * self.k_tile
                            + row * self.k_tile
                            + (col ^ ((row % 8) * 8))
                        )
                        src = activation.iterator.raw_ptr() + activation.layout(
                            (global_row, k_tile * self.k_tile + col)
                        )
                        prims.cp_async_shared_global(dst, src, 16, "cg")
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
            cute.arch.griddepcontrol_launch_dependents()
        else:
            mma_tid = tidx - self.compute_warp_base * cute.arch.WARP_SIZE
            compute_warp = warp_idx - self.compute_warp_base
            thr_mma = tiled_mma.get_slice(mma_tid)
            acc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self.output_tile, self.token_tile)),
                cutlass.Float32,
            )
            acc.fill(0.0)
            # RMS sums from the B fragments this warp already holds: every element of
            # its K quarter appears once, in lane l's registers for token l // 4 + 8 n.
            sq_acc = cute.make_rmem_tensor(
                cute.make_layout(self.token_tile // 8), cutlass.Float32
            )
            sq_acc.fill(0.0)
            k_phases = self.k_tile // (_COMPUTE_WARPS * _MMA_SHAPE[2])
            m_iters = self.output_tile // _MMA_SHAPE[0]
            n_iters = self.token_tile // _MMA_SHAPE[1]
            a_offsets = tuple(
                tuple(
                    _a_fragment_offset(lane_idx, compute_warp, m_iter, phase, self.k_tile)
                    for phase in range(k_phases)
                )
                for m_iter in range(m_iters)
            )
            b_offsets = tuple(
                tuple(
                    _b_pair_offset(
                        lane_idx, compute_warp, n_iter, pair, self.token_tile,
                        self.k_tile,
                    )
                    for pair in range(k_phases // 2)
                )
                for n_iter in range(n_iters)
            )
            a_stage_elems = self.output_tile * self.k_tile
            b_stage_elems = self.token_tile * self.k_tile
            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll_full=True):
                stage = k_tile % self.stages
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
                a_base = stage * a_stage_elems
                b_base = stage * b_stage_elems
                a_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sA, a_base + a_offsets[m_iter][phase])
                        for phase in range(k_phases)
                    )
                    for m_iter in range(m_iters)
                )
                b_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sB, b_base + b_offsets[n_iter][pair])
                        for pair in range(k_phases // 2)
                    )
                    for n_iter in range(n_iters)
                )
                if const_expr(self.norm):
                    for n_iter in cutlass.range_constexpr(n_iters):
                        part = sq_acc[n_iter]
                        for pair in cutlass.range_constexpr(k_phases // 2):
                            for i in cutlass.range_constexpr(4):
                                w = Uint32(b_registers[n_iter][pair][i])
                                lo = bits_float(w << 16)
                                hi = bits_float(w & Uint32(0xFFFF0000))
                                part = part + lo * lo
                                part = part + hi * hi
                        sq_acc[n_iter] = part
                for phase in cutlass.range_constexpr(k_phases):
                    for n_iter in cutlass.range_constexpr(n_iters):
                        packed_b = b_registers[n_iter][phase // 2]
                        b_offset = (phase % 2) * 2
                        for m_iter in cutlass.range_constexpr(m_iters):
                            acc_base = (m_iter * n_iters + n_iter) * 4
                            result = prims.mma_sync(
                                llvm.StructType.get_literal(
                                    [cutlass.Float32.mlir_type] * 4
                                ),
                                shape=_MMA_SHAPE,
                                layout_a=prims.MMALayout.ROW,
                                layout_b=prims.MMALayout.COL,
                                operand_a=[
                                    a_registers[m_iter][phase][i].ir_value()
                                    for i in range(4)
                                ],
                                operand_b=[
                                    packed_b[b_offset + i].ir_value() for i in range(2)
                                ],
                                operand_c=[
                                    acc[acc_base + i].ir_value() for i in range(4)
                                ],
                                multiplicand_a_ptx_type=prims.MMAType.BF16,
                                multiplicand_b_ptx_type=prims.MMAType.BF16,
                            )
                            for i in cutlass.range_constexpr(4):
                                acc[acc_base + i] = cutlass.Float32(
                                    llvm.extractvalue(
                                        cutlass.Float32.mlir_type, result, [i]
                                    )
                                )
                if const_expr(reuse_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_empty + stage)
                if stage == self.stages - 1:
                    full_phase = full_phase ^ 1
            for value in cutlass.range_constexpr(values_per_lane):
                atom = value % 4
                fragment = value // 4
                m_iter = fragment // n_iters
                n_iter = fragment % n_iters
                feature = lane_idx // 4 + 8 * (atom // 2) + m_iter * _MMA_SHAPE[0]
                token = 8 * n_iter + 2 * (lane_idx % 4) + atom % 2
                partials[
                    compute_warp * partial_stride
                    + _partial_offset(feature, token, self.token_tile)
                ] = acc[value]
            if const_expr(self.norm):
                for n_iter in cutlass.range_constexpr(self.token_tile // 8):
                    total = sq_acc[n_iter]
                    total = total + cute.arch.shuffle_sync_bfly(total, 1)
                    total = total + cute.arch.shuffle_sync_bfly(total, 2)
                    if lane_idx % 4 == 0:
                        sq[compute_warp * self.token_tile + lane_idx // 4 + 8 * n_iter] = total
            prims.barrier_cta_sync(1, thread_count=_COMPUTE_WARPS * 32)

        if warp_idx == self.compute_warp_base:  # noqa: SIM102
            output_base = output_tile_idx * self.output_tile
            token_base = token_tile_idx * self.token_tile
            if const_expr(self.act):
                half = self.output_tile // 2
                out_per_lane = half * self.token_tile // cute.arch.WARP_SIZE
                for value in cutlass.range_constexpr(out_per_lane):
                    linear = value * cute.arch.WARP_SIZE + lane_idx
                    feature = linear % half
                    token = linear // half
                    g = cutlass.Float32(0)
                    u = cutlass.Float32(0)
                    for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                        g = g + partials[
                            peer * partial_stride
                            + _partial_offset(feature, token, self.token_tile)
                        ]
                        u = u + partials[
                            peer * partial_stride
                            + _partial_offset(feature + half, token, self.token_tile)
                        ]
                    if const_expr(self.norm):
                        tot = cutlass.Float32(0)
                        for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                            tot = tot + sq[peer * self.token_tile + token]
                        rstd = rms_norm_rstd(tot, cutlass.Float32(self.eps))
                        g = g * rstd
                        u = u * rstd
                    g_bits = Uint32(gate_bf16_bits(g))
                    u_bits = Uint32(gate_bf16_bits(u))
                    s = table_load(table.iterator + g_bits)
                    u_f = bits_float(u_bits << 16)
                    res = s * (u_f + cutlass.Float32(0.0))
                    if token_base + token < cute.size(output, mode=[0]):
                        output[token_base + token, output_tile_idx * half + feature] = (
                            res.to(output.element_type)
                        )
            else:
                for value in cutlass.range_constexpr(values_per_lane):
                    linear = value * cute.arch.WARP_SIZE + lane_idx
                    feature = linear % self.output_tile
                    token = linear // self.output_tile
                    total = cutlass.Float32(0)
                    for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                        total = total + partials[
                            peer * partial_stride
                            + _partial_offset(feature, token, self.token_tile)
                        ]
                    if const_expr(self.norm):
                        tot = cutlass.Float32(0)
                        for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                            tot = tot + sq[peer * self.token_tile + token]
                        total = total * rms_norm_rstd(tot, cutlass.Float32(self.eps))
                    if token_base + token < cute.size(output, mode=[0]):
                        if const_expr(self.resid):
                            total = total + residual[
                                token_base + token, output_base + feature
                            ].to(cutlass.Float32)
                        output[token_base + token, output_base + feature] = total.to(
                            output.element_type
                        )


def _view(tensor, *, dynamic_m=False):
    view = from_dlpack(tensor.detach(), assumed_align=32)
    if dynamic_m:
        view = view.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
    return view


@functools.cache
def compile_drf_gemm(device_index: int, n: int, k: int, tactic: WarpSplitKTactic,
                     norm: bool, resid: bool, act: bool, eps: float):
    check_tactic(tactic, n, k)
    if norm and k != 7168:
        raise ValueError("norm fold: rms_norm_rstd divides by 7168")
    if act and tactic.output_tile // 2 * tactic.token_tile < 32:
        raise ValueError("act: half tile x token tile below a warp")
    dtype = _torch.bfloat16
    n_out = n // 2 if act else n
    with _torch.cuda.device(device_index):
        args = (
            _view(_torch.empty((n, k), device="cuda", dtype=dtype)),
            _view(_torch.empty((_MAX_M, k), device="cuda", dtype=dtype), dynamic_m=True),
            _view(_torch.empty((_MAX_M, n_out), device="cuda", dtype=dtype), dynamic_m=True),
            _view(_torch.empty((_MAX_M, n_out), device="cuda", dtype=dtype), dynamic_m=True),
            _view(_torch.empty((65536,), device="cuda", dtype=_torch.float32)),
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
        kernel = DrfWarpSplitKKernel(tactic, norm=norm, resid=resid, act=act, eps=eps)
        return cute_ext.compile(kernel, *args, stream)


_SILU_TABLES: dict = {}


def silu_table(device):
    """silu(g) for every BF16 bit pattern g, from vLLM's own silu_and_mul (up = 1),
    as FP32."""
    key = str(device)
    table = _SILU_TABLES.get(key)
    if table is None:
        bits = _torch.arange(65536, dtype=_torch.int32, device=device).to(_torch.int16)
        gate = bits.view(_torch.bfloat16).reshape(-1, 1)
        x = _torch.cat([gate.reshape(1, -1), _torch.ones_like(gate).reshape(1, -1)], dim=1)
        out = _torch.empty((1, 65536), dtype=_torch.bfloat16, device=device)
        _torch.ops._C.silu_and_mul(out, x)
        table = out.reshape(-1).float().contiguous()
        _SILU_TABLES[key] = table
    return table


def permute_gate_up(weight, output_tile):
    """[gate; up] rows -> per output tile: T/2 gate rows then the T/2 matching up
    rows (a new tensor)."""
    n2, k = weight.shape
    i = n2 // 2
    half = output_tile // 2
    g = weight[:i].reshape(i // half, half, k)
    u = weight[i:].reshape(i // half, half, k)
    return _torch.stack([g, u], dim=1).reshape(n2, k).contiguous()


class DrfGemm:
    """`out = f(x @ weight^T)` for M <= 32 with the options above, one compiled
    engine per M (compiled on the first eager call of that M)."""

    def __init__(self, weight, *, norm=False, resid=False, act=False, eps=1e-5,
                 tactic_fn=None):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.decode import warp_tactic

        self.n, self.k = weight.shape
        self.norm, self.resid, self.act, self.eps = norm, resid, act, float(eps)
        self.n_out = self.n // 2 if act else self.n
        self.sms = _torch.cuda.get_device_properties(weight.device).multi_processor_count
        self.tactic_fn = tactic_fn or (lambda m: warp_tactic(m, self.n, self.k, self.sms))
        self.weight = weight  # caller-prepared (permuted for act, folded for norm)
        self.codes: dict = {}
        self.table = silu_table(weight.device) if act else None
        self.weight_view = _view(weight)
        self._dummy = None

    def output_tile(self, m):
        return self.tactic_fn(m).output_tile

    def prepare(self, m):
        if m in self.codes or not 1 <= m <= _MAX_M:
            return
        t = self.tactic_fn(m)
        self.codes[m] = (t, compile_drf_gemm(
            self.weight.device.index, self.n, self.k, t, self.norm, self.resid,
            self.act, self.eps))

    def run(self, x, out, residual=None):
        m = x.shape[0]
        if m not in self.codes and not _torch.cuda.is_current_stream_capturing():
            self.prepare(m)
        _, code = self.codes[m]
        stream = _cuda.CUstream(_torch.cuda.current_stream(x.device).cuda_stream)
        res = residual if residual is not None else out
        table = self.table if self.table is not None else self._table_dummy(x.device)
        code(self.weight_view, _view(x, dynamic_m=True), _view(out, dynamic_m=True),
             _view(res, dynamic_m=True), _view(table), stream)
        return out

    def _table_dummy(self, device):
        if self._dummy is None:
            self._dummy = _torch.zeros(65536, dtype=_torch.float32, device=device)
        return self._dummy
