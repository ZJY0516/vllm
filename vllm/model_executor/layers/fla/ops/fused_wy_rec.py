# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused WY + recurrence kernel.
# Replaces 2 separate kernels (recompute_w_u + chunk_delta_h) with a single
# kernel that computes v_new on-the-fly per chunk. w and u are never
# materialized to HBM, saving ~16MB for typical configs (B=1, T=8192, H=4,
# K=128, V=128, bf16).
#
# Algorithm per chunk:
#   temp = v * beta - (k * beta * exp(g)) @ h
#   v_new = A_inv @ temp
#   h = h * exp(g_last) + k^T @ (v_new * exp(g_last - g))
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp
from .utils import use_cuda_graph


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fused_wy_rec_kernel(
    k,          # [B, T, Hg, K]
    v,          # [B, T, H, V]   (original v, not u)
    beta,       # [B, T, H]
    A_inv,      # [B, T, H, BT]
    g,          # [B, T, H]      (g_cumsum, float32)
    h,          # [B, NT, H, V, K] output: per-chunk states
    v_new,      # [B, T, H, V]   output: corrected values
    h0,         # [N, H, V, K]   initial state
    ht,         # [N, H, V, K]   final state
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Fused WY + recurrence: w and u computed on-the-fly, never in HBM."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # Precompute base pointers
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    # Initialize state [BV, 64] per K-block
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)

    # Load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

    # Main recurrence loop
    for i_t in range(NT):
        # ── Store current h state ────────────────────────────────────
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))

        # ── Load inputs for this chunk ───────────────────────────────
        p_beta = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)

        p_g = tl.make_block_ptr(
            g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_exp_g = exp(b_g)

        # ── Compute temp = v*beta - (k*beta*exp(g)) @ h (on-the-fly WY) ──
        p_v_in = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_temp = (tl.load(p_v_in, boundary_check=(0, 1)).to(tl.float32) * b_beta[:, None])

        # k_scaled = k * beta * exp(g), then k_scaled @ h
        p_k1 = tl.make_block_ptr(
            k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k1_scaled = (b_k1.to(tl.float32) * b_beta[:, None] * b_exp_g[:, None]).to(b_k1.dtype)
        b_temp -= tl.dot(b_k1_scaled, tl.trans(b_h1).to(b_k1_scaled.dtype))

        if K > 64:
            p_k2 = tl.make_block_ptr(
                k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))
            b_k2_scaled = (b_k2.to(tl.float32) * b_beta[:, None] * b_exp_g[:, None]).to(b_k2.dtype)
            b_temp -= tl.dot(b_k2_scaled, tl.trans(b_h2).to(b_k2_scaled.dtype))

        # ── v_new = A_inv @ temp ─────────────────────────────────────
        p_A = tl.make_block_ptr(
            A_inv + (bos * H + i_h) * BT,
            (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0),
        )
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_v_new = tl.dot(b_A, b_temp.to(b_A.dtype))

        # Store v_new
        p_v_new = tl.make_block_ptr(
            v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        # ── Update h state: h = h * exp(g_last) + k^T @ v_new_scaled ─
        last_idx = min((i_t + 1) * BT, T) - 1
        m_t = (i_t * BT + tl.arange(0, BT)) < T
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        b_v_decay = b_v_new * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
        b_v_decay = b_v_decay.to(k.dtype.element_ty)

        b_g_last_exp = exp(b_g_last)
        b_h1 *= b_g_last_exp
        if K > 64:
            b_h2 *= b_g_last_exp

        p_kT1 = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_h1 += tl.trans(tl.dot(tl.load(p_kT1, boundary_check=(0, 1)), b_v_decay))
        if K > 64:
            p_kT2 = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_h2 += tl.trans(tl.dot(tl.load(p_kT2, boundary_check=(0, 1)), b_v_decay))

    # ── Epilogue: store final state ──────────────────────────────────
    if STORE_FINAL_STATE:
        p_ht1 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht2 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1))


def fused_wy_recurrence(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A_inv: torch.Tensor,
    g_cumsum: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Fused WY + recurrence. w/u computed on-the-fly, never in HBM.

    Args:
        k: Keys [B, T, Hg, K]
        v: Values [B, T, H, V] (original v, NOT u)
        beta: Beta scalars [B, T, H]
        A_inv: Inverse matrix [B, T, H, BT] from fused_ckkt_solve
        g_cumsum: Cumulative gate [B, T, H] (float32)
        initial_state: [N, H, V, K] initial state
        output_final_state: Whether to output final state
        cu_seqlens: Cumulative sequence lengths [N+1]

    Returns:
        (h [B, NT, H, V, K], v_new [B, T, H, V], final_state or None)
    """
    B, T, Hg, K = k.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(v)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    fused_wy_rec_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        A_inv=A_inv,
        g=g_cumsum,
        h=h,
        v_new=v_new,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state
