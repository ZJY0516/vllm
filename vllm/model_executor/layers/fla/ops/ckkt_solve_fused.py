# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused cumsum + KKT + squaring-trick matrix inverse kernel.
# Replaces 3 separate kernels (cumsum, chunk_scaled_dot_kkt, solve_tril)
# with a single fused kernel that keeps intermediate A in registers.
#
# Algorithm: For strictly lower-triangular nilpotent A (A^64=0),
#   (I + A)^{-1} = (I - A)(I + A^2)(I + A^4)(I + A^8)(I + A^16)(I + A^32)
# This uses 10 tl.dot(64x64) instead of hierarchical 16x16 block elimination.
# ruff: noqa: E501

import os

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import exp
from .utils import input_guard

FLA_TRIL_PRECISION = os.environ.get("FLA_TRIL_PRECISION", "ieee")


@triton.heuristics(
    {
        "USE_G": lambda args: args["g_in"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def fused_ckkt_solve_kernel(
    k,
    beta,
    g_in,
    g_out,
    A_inv_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    # ── Stage 1: cumsum ──────────────────────────────────────────────
    if USE_G:
        p_g_in = tl.make_block_ptr(
            g_in + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g_raw = tl.load(p_g_in, boundary_check=(0,)).to(tl.float32)
        b_g = tl.cumsum(b_g_raw, axis=0)
        p_g_out = tl.make_block_ptr(
            g_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        tl.store(p_g_out, b_g.to(p_g_out.dtype.element_ty), boundary_check=(0,))

    # ── Stage 2: KKT ────────────────────────────────────────────────
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))

    if USE_G:
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)

    # Strictly lower-triangular mask
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0.0)

    # ── Stage 3: squaring-trick inverse ──────────────────────────────
    # (I + A)^{-1} = (I - A)(I + A^2)(I + A^4)(I + A^8)(I + A^16)(I + A^32)
    # A is strictly lower-triangular 64x64, so A^64 = 0 (nilpotent).
    bt_idx = tl.arange(0, BT)
    m_I = (bt_idx[:, None] == bt_idx[None, :]).to(tl.float32)

    # X = -A (negate for the (I - A) term)
    b_X = (-b_A).to(tl.bfloat16)
    b_I = m_I.to(tl.bfloat16)

    # X^2
    b_X2 = tl.dot(b_X, b_X, input_precision=DOT_PRECISION).to(tl.bfloat16)
    # (I - A)(I + A^2) = (I + X)(I + X^2)
    b_t12 = tl.dot(b_I + b_X, b_I + b_X2, input_precision=DOT_PRECISION)

    # X^4, X^8
    b_X4 = tl.dot(b_X2, b_X2, input_precision=DOT_PRECISION).to(tl.bfloat16)
    b_X8 = tl.dot(b_X4, b_X4, input_precision=DOT_PRECISION).to(tl.bfloat16)
    # (I + A^4)(I + A^8) = (I + X^4)(I + X^8)
    b_t48 = tl.dot(b_I + b_X4, b_I + b_X8, input_precision=DOT_PRECISION)

    # Combine: (I-A)(I+A^2) * (I+A^4)(I+A^8)
    b_t1248 = tl.dot(
        b_t12.to(tl.bfloat16), b_t48.to(tl.bfloat16),
        input_precision=DOT_PRECISION,
    )

    # X^16, X^32
    b_X16 = tl.dot(b_X8, b_X8, input_precision=DOT_PRECISION).to(tl.bfloat16)
    b_X32 = tl.dot(b_X16, b_X16, input_precision=DOT_PRECISION).to(tl.bfloat16)
    # (I + A^16)(I + A^32)
    b_t1632 = tl.dot(b_I + b_X16, b_I + b_X32, input_precision=DOT_PRECISION)

    # Final: (I+A)^{-1}
    b_Ai = tl.dot(
        b_t1248.to(tl.bfloat16), b_t1632.to(tl.bfloat16),
        input_precision=DOT_PRECISION,
    )

    # Store A_inv: same layout as solve_tril output [B, T, H, BT]
    p_Ai = tl.make_block_ptr(
        A_inv_out + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(
        p_Ai,
        b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@input_guard
def fused_ckkt_solve(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused cumsum + KKT + squaring-trick matrix inverse.

    Replaces the 3-kernel sequence: chunk_local_cumsum → chunk_scaled_dot_kkt
    → solve_tril with a single kernel that keeps intermediate A in registers.

    Args:
        k: Keys [B, T, Hg, K]
        g: Raw gate in log-space [B, T, H] (NOT cumsum'd)
        beta: Beta scalars [B, T, H]
        cu_seqlens: Cumulative sequence lengths [N+1] for varlen
        chunk_size: Chunk size (must be 64)
        output_dtype: dtype for A_inv output

    Returns:
        (A_inv [B, T, H, BT], g_cumsum [B, T, H])
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    assert BT == 64, "Squaring-trick inverse requires BT=64"

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A_inv = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    g_cumsum = torch.empty_like(g, dtype=torch.float32)

    fused_ckkt_solve_kernel[(NT, B * H)](
        k=k,
        beta=beta,
        g_in=g,
        g_out=g_cumsum,
        A_inv_out=A_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )
    return A_inv, g_cumsum
