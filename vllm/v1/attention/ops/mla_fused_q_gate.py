# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise-exact fusions around the MLA attention core.

`mla_q_concat_quant_pad` replaces three launches of the fp8-query MQA path
(copy the RoPE half beside the absorbed NoPE half, per-tensor static fp8
quantization, zero-padding the heads to the kernel's head count) with one.
`sigmoid_gate_mul_` replaces the two launches of the MLA output gate
(`attn_out * gate.sigmoid()`).

Both reproduce the unfused ops bit for bit: the same fp32 math in the same
order, the accurate `expf` (libdevice) rather than the approximate Triton
intrinsic, IEEE division that keeps subnormal results (see `_div_rn`) and the
same round-to-nearest-even casts.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _div_rn(x, y):
    """IEEE round-to-nearest fp32 `x / y`, as nvcc compiles it by default.
    libdevice's div_rn flushes subnormal results to zero here: a bf16 gate of
    -88 makes `1 / (1 + expf(88))` a subnormal that torch.sigmoid keeps."""
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mla_q_concat_quant_pad_kernel(
    nope_ptr,
    pe_ptr,
    out_ptr,
    scale_ptr,
    nope_stride_t,
    nope_stride_h,
    pe_stride_t,
    pe_stride_h,
    num_heads,
    padded_heads,
    FP8_MAX: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    is_real = heads < num_heads
    is_out = heads < padded_heads

    # As `fp8_reciprocal_scale` + `scaled_fp8_conversion<true>`: x * (1 / s),
    # clamped, cast with round-to-nearest-even. A zero scale maps to zero.
    # A [1, 1] block, so it broadcasts against the [BLOCK_H, dim] tiles.
    scale = tl.load(scale_ptr + tl.zeros([1, 1], dtype=tl.int32))
    inv_scale = tl.where(
        scale == 0.0, 0.0, _div_rn(tl.full([1, 1], 1.0, tl.float32), scale)
    )

    out_row = out_ptr + token * padded_heads * (NOPE_DIM + ROPE_DIM)
    out_head = out_row + heads[:, None] * (NOPE_DIM + ROPE_DIM)

    # Padding heads load zeros and so store fp8 zeros.
    nope_cols = tl.arange(0, NOPE_DIM)
    nope = tl.load(
        nope_ptr
        + token * nope_stride_t
        + heads[:, None] * nope_stride_h
        + nope_cols[None, :],
        mask=is_real[:, None],
        other=0.0,
    ).to(tl.float32)
    nope = tl.minimum(tl.maximum(nope * inv_scale, -FP8_MAX), FP8_MAX)
    tl.store(
        out_head + nope_cols[None, :],
        nope.to(out_ptr.dtype.element_ty),
        mask=is_out[:, None],
    )

    rope_cols = tl.arange(0, ROPE_DIM)
    rope = tl.load(
        pe_ptr
        + token * pe_stride_t
        + heads[:, None] * pe_stride_h
        + rope_cols[None, :],
        mask=is_real[:, None],
        other=0.0,
    ).to(tl.float32)
    rope = tl.minimum(tl.maximum(rope * inv_scale, -FP8_MAX), FP8_MAX)
    tl.store(
        out_head + NOPE_DIM + rope_cols[None, :],
        rope.to(out_ptr.dtype.element_ty),
        mask=is_out[:, None],
    )


def mla_q_concat_quant_pad(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    scale: torch.Tensor,
    padded_heads: int,
) -> torch.Tensor:
    """`[ql_nope | q_pe]` quantized to fp8 with the static per-tensor `scale`,
    in a new contiguous `[T, padded_heads, L + R]` buffer whose heads past
    `ql_nope.shape[1]` are zero.

    Matches `torch.cat((ql_nope, q_pe), -1)` -> `ops.scaled_fp8_quant(...,
    scale)` -> zero-pad heads, bit for bit. Inputs may be strided views as long
    as their last dimension is contiguous.
    """
    num_tokens, num_heads, nope_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    assert q_pe.shape[:2] == (num_tokens, num_heads)
    assert ql_nope.stride(-1) == 1 and q_pe.stride(-1) == 1
    assert padded_heads >= num_heads
    assert scale.numel() == 1 and scale.dtype == torch.float32
    assert triton.next_power_of_2(nope_dim) == nope_dim
    assert triton.next_power_of_2(rope_dim) == rope_dim
    fp8_dtype = current_platform.fp8_dtype()
    out = ql_nope.new_empty(
        (num_tokens, padded_heads, nope_dim + rope_dim), dtype=fp8_dtype
    )
    if num_tokens == 0:
        return out
    block_h = 16
    _mla_q_concat_quant_pad_kernel[(num_tokens, triton.cdiv(padded_heads, block_h))](
        ql_nope,
        q_pe,
        out,
        scale,
        ql_nope.stride(0),
        ql_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        num_heads,
        padded_heads,
        FP8_MAX=torch.finfo(fp8_dtype).max,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


@triton.jit
def _sigmoid_gate_mul_kernel(
    x_ptr,
    gate_ptr,
    x_stride,
    gate_stride,
    num_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < num_cols
    gate = tl.load(gate_ptr + row * gate_stride + cols, mask=mask, other=0.0)
    # torch.sigmoid on a 16-bit tensor: 1 / (1 + expf(-g)) in fp32, rounded
    # back to the tensor dtype; then the mul in fp32, rounded again.
    gate = gate.to(tl.float32)
    sig = _div_rn(tl.full([BLOCK], 1.0, tl.float32), 1.0 + tldevice.exp(-gate))
    sig = sig.to(x_ptr.dtype.element_ty).to(tl.float32)
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)
    out = x.to(tl.float32) * sig
    tl.store(x_ptr + row * x_stride + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate_mul_(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """In place `x *= gate.sigmoid()` for 2-D 16-bit `x` and `gate` of the same
    dtype and shape; bit for bit the two eager ops, in one launch."""
    assert x.dim() == 2 and x.shape == gate.shape
    assert x.dtype == gate.dtype and x.dtype in (torch.bfloat16, torch.float16)
    assert x.stride(1) == 1 and gate.stride(1) == 1
    num_rows, num_cols = x.shape
    if num_rows == 0 or num_cols == 0:
        return x
    block = 2048
    _sigmoid_gate_mul_kernel[(num_rows, triton.cdiv(num_cols, block))](
        x,
        gate,
        x.stride(0),
        gate.stride(0),
        num_cols,
        BLOCK=block,
        num_warps=2,
    )
    return x
