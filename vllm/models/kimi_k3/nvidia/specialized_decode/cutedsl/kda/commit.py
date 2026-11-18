# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM commit of an FP32 KDA state: one CTA per (head, request, layer).

Folds the n accepted tokens of a verify window into the checkpoint as the stock
FP32 commit (`_commit_kda_state_kernel`) does, in its association:

    S_n = S_0 * D_n + sum_{t<n} (u_t * kn_t) * P_t,   P_t = prod_{t<s<n} d_s

(`u_t` the verify's corrections, `kn_t` the normalized recorded keys, `d_t` the
per-key decays, `D_n = P_{-1}`), and with prefix caching (`align`) also the
boundary state S_m of the window's first m tokens.

Thread c < 128 rebuilds key channel c's decays, normalized keys and suffix products
into shared memory; then all 8 warps fold the head's [128, 128] state from global
memory to global memory, thread 2 i + h taking keys 8 c + 4 h + [0, 4) of row i,
so each byte of the state is read once and written once (the stock FP32 kernel
splits a head over four CTAs that each rebuild the key operands)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from ..common.host import architecture, dynamic_fake
from ..common.primitives import (
    alloc,
    fma,
    launch,
    load_f32x4,
    load_words,
    prefetch_bytes,
    store_f32x4,
    warp_sum,
)
from .kernel import triton_exp, triton_sigmoid

K = 128
THREADS = 256
CHUNKS = 16
PAGE_BYTES = K * K * 4


class KdaCommitF32:
    """Args as the stock CuTe commit's `KdaCommit` (base addresses, strides,
    a_log [L, H], dt_bias [L, H, K], source / commit length / destination /
    boundary slot / boundary length per request, lower bound, null slot, stream);
    the state is FP32 [H, V, K] per slot."""

    def __init__(
        self,
        heads: int,
        window: int,
        align: bool = False,
        ahead: int = 0,
        indexed: bool = False,
    ):
        assert 1 <= window <= 8
        self.heads, self.window, self.align = heads, window, align
        # ``indexed``: a trailing record_rows [B] operand gives each request's record
        # slot (deferred RecoverSSM keeps the records per request slot, not step row).
        self.indexed = indexed
        # Resident CTAs: L2 prefetch of the next wave's pages (0: off; measured
        # 372 -> 397 us at TP1 R128, so the commit runs without it).
        self.ahead = ahead

    @cute.jit
    def __call__(self, *args):
        grid = (self.heads, args[8].shape[0], args[0].shape[0])
        launch(self.kernel, args, grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        state_addrs, corr_addrs, kg_addrs = args[:3]
        state_strides, corr_strides, kg_strides, a_log, dt_bias = args[3:8]
        sources, commit_lens, destinations, boundaries, boundary_lens = args[8:13]
        lower_bound, null_slot = args[13:15]
        Q: cutlass.Constexpr = self.window
        ALIGN: cutlass.Constexpr = self.align
        tid, _, _ = cute.arch.thread_idx()
        head, request, layer = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        smem = cutlass.utils.SmemAllocator()
        keys = alloc(smem, Float32, (Q, K))  # kn_t
        suffix = alloc(smem, Float32, (Q + 1, K))  # P_t; row Q: D_n
        suffix_b = alloc(smem, Float32, (Q + 1, K))  # the same for S_m (align)
        norms = alloc(smem, Float32, (Q, 4))

        cute.arch.griddepcontrol_wait()
        n = commit_lens[request]
        source = Int64(sources[request])
        destination = Int64(destinations[request])
        live = (n > 0) & (source > Int64(null_slot)) & (destination > Int64(null_slot))
        m = Int32(0)
        boundary = Int64(null_slot)
        if cutlass.const_expr(ALIGN):
            m = boundary_lens[request]
            boundary = Int64(boundaries[request])
        boundary_live = (m > 0) & (boundary > Int64(null_slot))
        if cutlass.const_expr(self.ahead > 0):  # noqa: SIM102
            if tid == 0:
                _, requests, layers = cute.arch.grid_dim()
                item = head + self.heads * (request + requests * layer) + self.ahead
                if item < self.heads * requests * layers:
                    h = item % self.heads
                    r = (item // self.heads) % requests
                    lay = item // (self.heads * requests)
                    slot = Int64(sources[r])
                    if slot > Int64(null_slot):
                        address = Int64(state_addrs[lay])
                        address += slot * Int64(state_strides[lay]) * 4
                        address += Int64(h) * PAGE_BYTES
                        pointer = cute.make_ptr(
                            Float32, address, cute.AddressSpace.gmem, assumed_align=16
                        )
                        prefetch_bytes(pointer, PAGE_BYTES)
        if live:
            state_base = Int64(state_addrs[layer])
            stride = Int64(state_strides[layer]) * 4  # bytes per slot
            page = Int64(head) * PAGE_BYTES
            row, half = tid // 2, tid % 2

            def state_page(slot):
                address = state_base + slot * stride + page
                pointer = cute.make_ptr(
                    Float32, address, cute.AddressSpace.gmem, assumed_align=16
                )
                return cute.make_tensor(
                    pointer, cute.make_layout((K, K), stride=(K, 1))
                )

            src = state_page(source)
            corr_ptr = cute.make_ptr(
                Float32, corr_addrs[layer], cute.AddressSpace.gmem, assumed_align=16
            )
            kg_ptr = cute.make_ptr(
                cutlass.BFloat16,
                kg_addrs[layer],
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            record = Int64(request)
            if cutlass.const_expr(self.indexed):
                record = Int64(args[15][request])
            corr_slot = corr_ptr + record * Int64(corr_strides[layer])
            kg_slot = kg_ptr + record * Int64(kg_strides[layer])
            corr = cute.make_tensor(
                corr_slot + head * Q * K, cute.make_layout((Q, K), stride=(K, 1))
            )
            kg = cute.make_tensor(
                kg_slot + head * Q * 2 * K,
                cute.make_layout((Q, 2 * K), stride=(2 * K, 1)),
            )
            # This thread's row of corrections u_t (0 past n).
            u = cute.make_rmem_tensor(Q, Float32)
            for t in cutlass.range_constexpr(Q):
                u[t] = Float32(0)
                if t < n:
                    u[t] = corr[t, row]

            channel = tid % K
            kv = cute.make_rmem_tensor(Q, Float32)
            decays = cute.make_rmem_tensor(Q, Float32)
            if tid < K:
                bias = Float32(dt_bias[layer, head, channel])
                a = triton_exp(Float32(a_log[layer, head]))
                for t in cutlass.range_constexpr(Q):
                    kv[t], decays[t] = Float32(0), Float32(1)
                    if t < n:
                        kv[t] = Float32(kg[t, channel])
                        raw_g = Float32(kg[t, K + channel])
                        gate = lower_bound * triton_sigmoid(a * (raw_g + bias))
                        decays[t] = triton_exp(gate)
                for t in cutlass.range_constexpr(Q):
                    total = warp_sum(kv[t] * kv[t])
                    if lane == 0:
                        norms[t, warp] = total
            cute.arch.barrier()
            if tid < K:
                fd, fd_b = Float32(1), Float32(1)
                for r in cutlass.range_constexpr(Q):
                    t = Q - 1 - r  # reverse: the stock accumulation order
                    total = (norms[t, 0] + norms[t, 1]) + (norms[t, 2] + norms[t, 3])
                    keys[t, channel] = kv[t] * cute.math.rsqrt(total + Float32(1e-6))
                    suffix[t, channel] = fd
                    fd = fd * decays[t]  # 1 past n
                    if cutlass.const_expr(ALIGN):  # tokens past m add nothing
                        suffix_b[t, channel] = Float32(0)
                        if t < m:
                            suffix_b[t, channel] = fd_b
                            fd_b = fd_b * decays[t]
                suffix[Q, channel] = fd
                if cutlass.const_expr(ALIGN):
                    suffix_b[Q, channel] = fd_b
            cute.arch.barrier()

            dst = state_page(destination)
            if cutlass.const_expr(ALIGN):
                dst_b = state_page(boundary)
            load_shared = lambda tensor, t, column: load_words(
                tensor.iterator + t * K + column,
                count=4,
                shared=True,
                mutable=True,
                result_type=Float32,
                word_type="f32",
            )
            for c in cutlass.range_constexpr(CHUNKS):
                column = 8 * c + 4 * half
                initial = load_f32x4(src.iterator + row * K + column)
                final = [Float32(0) for _ in range(4)]
                final_b = [Float32(0) for _ in range(4)]
                for r in cutlass.range_constexpr(Q):
                    t = Q - 1 - r
                    kn = load_shared(keys, t, column)
                    p = load_shared(suffix, t, column)
                    if cutlass.const_expr(ALIGN):
                        p_b = load_shared(suffix_b, t, column)
                    for e in cutlass.range_constexpr(4):
                        update = u[t] * kn[e]  # 0 past n
                        final[e] = fma(update, p[e], final[e])
                        if cutlass.const_expr(ALIGN):
                            final_b[e] = fma(update, p_b[e], final_b[e])
                if cutlass.const_expr(ALIGN):  # first: the final state may share it
                    decay_m = load_shared(suffix_b, Q, column)
                    values_b = [
                        fma(initial[e], decay_m[e], final_b[e]) for e in range(4)
                    ]
                    store_f32x4(
                        dst_b.iterator + row * K + column, values_b, boundary_live
                    )
                decay_n = load_shared(suffix, Q, column)
                values = [fma(initial[e], decay_n[e], final[e]) for e in range(4)]
                store_f32x4(dst.iterator + row * K + column, values)
        cute.arch.griddepcontrol_launch_dependents()


class KdaFlushF32:
    """The deferred RecoverSSM flush in one launch, per (head, request, layer): the
    flush plan (`_prepare_flush_plan_kernel`: n = the pending count of the request's
    record slot, clamped to [0, window], live for a state slot past the null slot),
    the conv compaction of the head's q, k and v channels
    (`_compact_conv_state_kernel`: the `history` taps from n - 1 moved to 0) and the
    fold of `KdaCommitF32` in place (its operands, association and stores): the same
    values in one kernel instead of three. The CTA's state page is prefetched to L2
    at entry; the conv taps are loaded before the fold and stored after it. Args:
    KdaCommitF32's base
    addresses, strides, a_log and dt_bias, then the state slots, record rows and
    pending counts [B], the conv state base addresses and block / dim / token
    strides [L], the lower bound and the null slot; `grouped`: layers of several
    KDA cache groups in one launch, a trailing slot_addrs [L] operand giving each
    layer's state-slot tensor ([B] int32, unit stride; `slots` sizes the grid)."""

    def __init__(self, heads, window, history, conv_dtype, grouped=False):
        assert 1 <= window <= 8 and 1 <= history
        self.heads, self.window, self.history = heads, window, history
        self.conv_dtype, self.grouped = conv_dtype, grouped

    @cute.jit
    def __call__(self, *args):
        grid = (self.heads, args[8].shape[0], args[0].shape[0])
        launch(self.kernel, args, grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        state_addrs, corr_addrs, kg_addrs = args[:3]
        state_strides, corr_strides, kg_strides, a_log, dt_bias = args[3:8]
        slots, record_rows, pending = args[8:11]
        conv_addrs, conv_blocks, conv_dims, conv_tokens = args[11:15]
        lower_bound, null_slot = args[15:17]
        Q: cutlass.Constexpr = self.window
        H: cutlass.Constexpr = self.history
        tid, _, _ = cute.arch.thread_idx()
        head, request, layer = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        smem = cutlass.utils.SmemAllocator()
        keys = alloc(smem, Float32, (Q, K))  # kn_t
        suffix = alloc(smem, Float32, (Q + 1, K))  # P_t; row Q: D_n
        norms = alloc(smem, Float32, (Q, 4))

        cute.arch.griddepcontrol_wait()
        source = Int64(0)
        if cutlass.const_expr(self.grouped):
            slot_at = Int64(args[17][layer]) + Int64(request) * 4
            slot_ptr = cute.make_ptr(Int32, slot_at, cute.AddressSpace.gmem)
            source = Int64(cute.make_tensor(slot_ptr, cute.make_layout(1))[0])
        else:
            source = Int64(slots[request])
        record = record_rows[request]
        n = Int32(0)
        if record >= 0:
            n = pending[record]
        if n > Q:
            n = Int32(Q)
        if n < 0:
            n = Int32(0)
        live = (n > 0) & (source > Int64(null_slot))
        if live:
            state_base = Int64(state_addrs[layer])
            stride = Int64(state_strides[layer]) * 4  # bytes per slot
            page = Int64(head) * PAGE_BYTES
            row, half = tid // 2, tid % 2
            address = state_base + source * stride + page
            pointer = cute.make_ptr(
                Float32, address, cute.AddressSpace.gmem, assumed_align=16
            )
            src = cute.make_tensor(pointer, cute.make_layout((K, K), stride=(K, 1)))
            if tid == 0:
                prefetch_bytes(src.iterator, PAGE_BYTES)

            # The conv compaction of this head's q, k and v channels: loads now,
            # stores after the state's.
            conv_bytes: cutlass.Constexpr = self.conv_dtype.width // 8
            dim: cutlass.Constexpr = self.heads * K
            parts: cutlass.Constexpr = -(-3 * K // THREADS)
            conv_base = Int64(conv_addrs[layer])
            conv_block = Int64(conv_blocks[layer])
            conv_dim = Int64(conv_dims[layer])
            conv_token = Int64(conv_tokens[layer]) * conv_bytes
            conv_taps = cute.make_rmem_tensor((parts, H), self.conv_dtype)
            conv_rows = []
            for i in cutlass.range_constexpr(parts):
                conv_part = tid + THREADS * i
                conv_safe = conv_part % (3 * K)
                conv_channel = (conv_safe // K) * dim + head * K + conv_safe % K
                conv_row = conv_base + (
                    source * conv_block + Int64(conv_channel) * conv_dim
                ) * conv_bytes
                conv_rows.append(conv_row)
                if conv_part < 3 * K:
                    for h in cutlass.range_constexpr(H):
                        conv_at = conv_row + Int64(n - 1 + h) * conv_token
                        conv_ptr = cute.make_ptr(
                            self.conv_dtype, conv_at, cute.AddressSpace.gmem
                        )
                        conv_taps[i, h] = cute.make_tensor(conv_ptr, cute.make_layout(1))[0]

            corr_ptr = cute.make_ptr(
                Float32, corr_addrs[layer], cute.AddressSpace.gmem, assumed_align=16
            )
            kg_ptr = cute.make_ptr(
                cutlass.BFloat16,
                kg_addrs[layer],
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            record64 = Int64(record)
            corr_slot = corr_ptr + record64 * Int64(corr_strides[layer])
            kg_slot = kg_ptr + record64 * Int64(kg_strides[layer])
            corr = cute.make_tensor(
                corr_slot + head * Q * K, cute.make_layout((Q, K), stride=(K, 1))
            )
            kg = cute.make_tensor(
                kg_slot + head * Q * 2 * K,
                cute.make_layout((Q, 2 * K), stride=(2 * K, 1)),
            )
            u = cute.make_rmem_tensor(Q, Float32)
            for t in cutlass.range_constexpr(Q):
                u[t] = Float32(0)
                if t < n:
                    u[t] = corr[t, row]

            channel = tid % K
            kv = cute.make_rmem_tensor(Q, Float32)
            decays = cute.make_rmem_tensor(Q, Float32)
            if tid < K:
                bias = Float32(dt_bias[layer, head, channel])
                a = triton_exp(Float32(a_log[layer, head]))
                for t in cutlass.range_constexpr(Q):
                    kv[t], decays[t] = Float32(0), Float32(1)
                    if t < n:
                        kv[t] = Float32(kg[t, channel])
                        raw_g = Float32(kg[t, K + channel])
                        gate = lower_bound * triton_sigmoid(a * (raw_g + bias))
                        decays[t] = triton_exp(gate)
                for t in cutlass.range_constexpr(Q):
                    total = warp_sum(kv[t] * kv[t])
                    if lane == 0:
                        norms[t, warp] = total
            cute.arch.barrier()
            if tid < K:
                fd = Float32(1)
                for r in cutlass.range_constexpr(Q):
                    t = Q - 1 - r  # reverse: the stock accumulation order
                    total = (norms[t, 0] + norms[t, 1]) + (norms[t, 2] + norms[t, 3])
                    keys[t, channel] = kv[t] * cute.math.rsqrt(total + Float32(1e-6))
                    suffix[t, channel] = fd
                    fd = fd * decays[t]  # 1 past n
                suffix[Q, channel] = fd
            cute.arch.barrier()

            load_shared = lambda tensor, t, column: load_words(
                tensor.iterator + t * K + column,
                count=4,
                shared=True,
                mutable=True,
                result_type=Float32,
                word_type="f32",
            )
            for c in cutlass.range_constexpr(CHUNKS):
                column = 8 * c + 4 * half
                initial = load_f32x4(src.iterator + row * K + column)
                final = [Float32(0) for _ in range(4)]
                for r in cutlass.range_constexpr(Q):
                    t = Q - 1 - r
                    kn = load_shared(keys, t, column)
                    p = load_shared(suffix, t, column)
                    for e in cutlass.range_constexpr(4):
                        update = u[t] * kn[e]  # 0 past n
                        final[e] = fma(update, p[e], final[e])
                decay_n = load_shared(suffix, Q, column)
                values = [fma(initial[e], decay_n[e], final[e]) for e in range(4)]
                store_f32x4(src.iterator + row * K + column, values)
            for i in cutlass.range_constexpr(parts):
                if tid + THREADS * i < 3 * K:
                    for h in cutlass.range_constexpr(H):
                        conv_at = conv_rows[i] + Int64(h) * conv_token
                        conv_ptr = cute.make_ptr(
                            self.conv_dtype, conv_at, cute.AddressSpace.gmem
                        )
                        conv_dst = cute.make_tensor(conv_ptr, cute.make_layout(1))
                        conv_dst[0] = conv_taps[i, h]
        cute.arch.griddepcontrol_launch_dependents()


@cache
def _flush_compiled(heads: int, window: int, history: int, arch: str, param_dtype, conv_dtype):
    from cutlass.cute.runtime import make_fake_stream

    i32, i64 = torch.int32, torch.int64
    fakes = (
        *(dynamic_fake(1, i64) for _ in range(6)),  # base addresses, strides [L]
        dynamic_fake(2, param_dtype),  # A_log [L, H]
        dynamic_fake(3, param_dtype),  # dt_bias [L, H, K]
        *(dynamic_fake(1, i32) for _ in range(3)),  # slots, record rows [B], pending
        *(dynamic_fake(1, i64) for _ in range(4)),  # conv addresses, strides [L]
    )
    cute_conv = {torch.float32: Float32, torch.bfloat16: cutlass.BFloat16}[conv_dtype]
    return cute.compile(
        KdaFlushF32(heads, window, history, cute_conv, grouped=True),
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        dynamic_fake(1, i64),  # slot_addrs [L]
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch={arch} --enable-tvm-ffi",
    )


def merge_enabled() -> bool:
    """K31_S24FM=1: a step's deferred flushes of every KDA cache group in one launch
    (KdaFlushF32), for the heads per rank in K31_S24FL_HEADS (default TP8's 12)."""
    import os

    return os.environ.get("K31_S24FM") == "1"


def merge_heads() -> set:
    import os

    spec = os.environ.get("K31_S24FL_HEADS", "12")
    return {int(h) for h in spec.split(",") if h.strip()}


_MERGED: dict = {}  # (context ids, slot tensor addresses) -> merged operands


def _merge_key(contexts, slots):
    return tuple(id(c) for c in contexts), tuple(s.data_ptr() for s in slots)


def prepare_merged(contexts, slots) -> bool:
    """Concatenate the per-layer flush operands of `contexts` (the KDA cache groups
    of one step, with their [B] state-slot tensors `slots`) and compile the grouped
    KdaFlushF32: at metadata build (outside graph capture), once per combination."""
    key = _merge_key(contexts, slots)
    if key in _MERGED:
        return _MERGED[key] is not None
    first = contexts[0]
    state = first.checkpoints[0]
    same = all(
        c.checkpoints[0].shape[1:] == state.shape[1:]
        and c.checkpoints[0].dtype == state.dtype
        and supported(c.checkpoints[0], c.lower_bound)
        and c.spec_query_len == first.spec_query_len
        and c.conv_history_len == first.conv_history_len
        and c.conv_states[0].dtype == first.conv_states[0].dtype
        and c.conv_states[0].dtype in (torch.float32, torch.bfloat16)
        and c.A_log.dtype == first.A_log.dtype
        and c.lower_bound == first.lower_bound
        for c in contexts
    )
    if (
        not same
        or state.shape[1] not in merge_heads()
        or any(s.dim() != 1 or s.stride(0) != 1 for s in slots)
    ):
        _MERGED[key] = None
        return False
    cat = lambda name: torch.cat([getattr(c, name) for c in contexts]).contiguous()
    counts = [len(c.checkpoints) for c in contexts]
    addrs = [s.data_ptr() for s, n in zip(slots, counts) for _ in range(n)]
    operands = tuple(
        cat(name)
        for name in (
            "state_base_addrs",
            "correction_cache_base_addrs",
            "kg_cache_base_addrs",
            "state_block_strides",
            "correction_cache_block_strides",
            "kg_cache_block_strides",
            "A_log",
            "dt_bias",
        )
    )
    convs = tuple(
        cat(name)
        for name in (
            "conv_state_base_addrs",
            "conv_state_block_strides",
            "conv_state_dim_strides",
            "conv_state_token_strides",
        )
    )
    slot_addrs = torch.tensor(addrs, dtype=torch.int64, device=state.device)
    compiled = _flush_compiled(
        state.shape[1],
        first.spec_query_len,
        first.conv_history_len,
        architecture(state.device),
        first.A_log.dtype,
        first.conv_states[0].dtype,
    )
    _MERGED[key] = (compiled, operands, convs, slot_addrs, first.lower_bound)
    return True


def flush_merged(contexts, slots, record_rows, pending) -> bool:
    """The grouped KdaFlushF32 over every layer of `contexts` (prepared by
    `prepare_merged`), request b's record slot `record_rows[b]`; False if not
    prepared."""
    entry = _MERGED.get(_merge_key(contexts, slots))
    if entry is None:
        return False
    compiled, operands, convs, slot_addrs, lower_bound = entry
    batch = slots[0].shape[0]
    compiled(
        *operands,
        slots[0],
        record_rows[:batch],
        pending,
        *convs,
        lower_bound,
        NULL_BLOCK_ID,
        slot_addrs,
    )
    return True


@cache
def _compiled(
    heads: int, window: int, arch: str, param_dtype, align: bool, indexed=False
):
    from cutlass.cute.runtime import make_fake_stream

    i32, i64 = torch.int32, torch.int64
    fakes = (
        *(dynamic_fake(1, i64) for _ in range(6)),  # base addresses, strides [L]
        dynamic_fake(2, param_dtype),  # A_log [L, H]
        dynamic_fake(3, param_dtype),  # dt_bias [L, H, K]
        *(dynamic_fake(1, i32) for _ in range(5)),  # per-request plan [B]
    )
    rows = (dynamic_fake(1, i32),) if indexed else ()
    return cute.compile(
        KdaCommitF32(heads, window, align, indexed=indexed),
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        *rows,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch={arch} --enable-tvm-ffi",
    )


# (head, request, layer) items of a launch from which KdaCommitF32 beats the
# Triton commit (GB300, q5, us per layer Triton / ours over 69 layers: TP8 R1
# 0.89 / 0.66, R2 1.52 / 1.13; TP1 R1 4.99 / 3.34; over 8 layers: TP8 R1 1.63 / 1.88,
# R4 3.50 / 2.92, R16 10.1 / 7.6; over 2: TP1 R128 572 / 360). A Kimi K3.1 rank
# commits its 69 KDA layers in one launch, so this holds at every batch.
COMMIT_ITEMS = 384


def use_commit_f32(heads: int, batch: int, layers: int = 1) -> bool:
    return heads * batch * layers >= COMMIT_ITEMS


def supported(state: torch.Tensor, lower_bound: float | None) -> bool:
    """Whether `commit_f32` folds this checkpoint cache (FP32 [slot, H, 128, 128]
    pages, contiguous per slot, on SM10x) for a bounded gate (the stock softplus
    gate of `gate_lower_bound=None` is not implemented)."""
    return (
        lower_bound is not None
        and state.dtype == torch.float32
        and state.is_cuda
        and torch.cuda.get_device_capability(state.device)[0] == 10
        and state.shape[2:] == (K, K)
        and state.stride()[1:] == (K * K, K, 1)
    )


def compile_commit_f32(
    state: torch.Tensor, window: int, params, align: bool, indexed: bool = False
):
    """Compile before serving (the commit runs eagerly after sampling)."""
    arch = architecture(state.device)
    _compiled(state.shape[1], window, arch, params, align, indexed)


def commit_f32(
    context,
    state_indices: torch.Tensor,
    batch: int,
    align: bool,
    record_rows: torch.Tensor | None = None,
):
    """Fold a prepared commit plan of `context` (a `KDARecoverSSMCommitContext`)
    into every layer's FP32 state in one launch on the current stream; the records
    of request b are in slot `record_rows[b]` (default: b)."""
    state = context.checkpoints[0]
    compiled = _compiled(
        state.shape[1],
        context.spec_query_len,
        architecture(state.device),
        context.A_log.dtype,
        align,
        record_rows is not None,
    )
    rows = () if record_rows is None else (record_rows[:batch],)
    compiled(
        context.state_base_addrs,
        context.correction_cache_base_addrs,
        context.kg_cache_base_addrs,
        context.state_block_strides,
        context.correction_cache_block_strides,
        context.kg_cache_block_strides,
        context.A_log,
        context.dt_bias,
        state_indices[:batch],
        context.commit_lens[:batch],
        context.final_state_indices[:batch],
        context.boundary_state_indices[:batch],
        context.boundary_recovery_lens[:batch],
        context.lower_bound,
        NULL_BLOCK_ID,
        *rows,
    )
