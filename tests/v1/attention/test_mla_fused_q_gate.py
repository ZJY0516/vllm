# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused MLA query prep and output gate match the unfused ops bit for bit."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.attention.ops.mla_fused_q_gate import (
    mla_q_absorb_quant_pad,
    mla_q_concat_quant_pad,
    sigmoid_gate_mul_,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="needs a CUDA GPU with fp8",
)


def _padded_heads(num_heads: int) -> int:
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        trtllm_gen_query_heads,
    )

    return trtllm_gen_query_heads(num_heads)


def _unfused_q(ql_nope, q_pe, scale, padded_heads):
    """The fp8-query MQA path: concat in place, static fp8 quant, pad heads."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        _pad_query_heads,
    )

    num_tokens, num_heads, nope_dim = ql_nope.shape
    cat = ql_nope.new_empty((num_tokens, num_heads, nope_dim + q_pe.shape[-1]))
    cat[..., :nope_dim].copy_(ql_nope)
    cat[..., nope_dim:].copy_(q_pe)
    q, _ = ops.scaled_fp8_quant(cat.reshape(num_tokens, -1), scale)
    q = q.view(cat.shape)
    if padded_heads != num_heads:
        q = _pad_query_heads(q, padded_heads)
    return q


def _native_q(ql_nope, q_pe, scale, padded_heads):
    """QuantFP8.forward_native's math, then zero padding heads."""
    fp8 = current_platform.fp8_dtype()
    finfo = torch.finfo(fp8)
    cat = torch.cat((ql_nope, q_pe), dim=-1).float()
    q = (cat * scale.float().reciprocal()).clamp(finfo.min, finfo.max).to(fp8)
    out = torch.zeros(
        (q.shape[0], padded_heads, q.shape[2]), dtype=fp8, device=q.device
    )
    out[:, : q.shape[1]] = q
    return out


@pytest.mark.parametrize("num_tokens", [1, 7, 513, 2048])
@pytest.mark.parametrize("num_heads", [96, 64, 16])
@pytest.mark.parametrize("scale", [1.0, 0.37, 3.5, 0.0])
@pytest.mark.parametrize("magnitude", [1.0, 300.0])
def test_q_concat_quant_pad_bitwise(num_tokens, num_heads, scale, magnitude):
    torch.manual_seed(num_tokens * 131 + num_heads)
    device = "cuda"
    nope_dim, rope_dim, qk_nope = 512, 64, 128
    # As in forward_impl: the absorbed NoPE half is a column slice of the
    # token-major concat buffer, the RoPE half a column slice of q.
    cat_buf = torch.randn(
        num_tokens, num_heads, nope_dim + rope_dim, device=device
    ).mul_(magnitude)
    ql_nope = cat_buf.to(torch.bfloat16)[..., :nope_dim]
    q = (
        torch.randn(num_tokens, num_heads, qk_nope + rope_dim, device=device)
        .mul_(magnitude)
        .to(torch.bfloat16)
    )
    q_pe = q[..., qk_nope:]
    scale_t = torch.tensor([scale], dtype=torch.float32, device=device)
    padded = _padded_heads(num_heads)

    fused = mla_q_concat_quant_pad(ql_nope, q_pe, scale_t, padded)
    assert fused.shape == (num_tokens, padded, nope_dim + rope_dim)
    assert fused.is_contiguous()

    ref = _unfused_q(ql_nope, q_pe, scale_t, padded)
    assert torch.equal(fused.view(torch.uint8), ref.view(torch.uint8))
    if scale != 0.0:
        native = _native_q(ql_nope, q_pe, scale_t, padded)
        assert torch.equal(fused.view(torch.uint8), native.view(torch.uint8))
    # Padding heads are exact zeros.
    assert not fused[:, num_heads:].view(torch.uint8).any()


def test_q_concat_quant_pad_special_values():
    device = "cuda"
    fp8 = current_platform.fp8_dtype()
    specials = torch.tensor(
        [0.0, -0.0, 448.0, -448.0, 449.0, 464.0, 1e6, -1e6, 2.0**-9, 2.0**-10]
        + [2.0**-6 * 1.0625, 2.0**-130, float("inf"), float("-inf")],
        device=device,
    ).to(torch.bfloat16)
    ql_nope = specials.repeat(512 // specials.numel() + 1)[:512]
    ql_nope = ql_nope.view(1, 1, 512).expand(3, 96, 512).contiguous()
    q_pe = specials.repeat(64 // specials.numel() + 1)[:64]
    q_pe = q_pe.view(1, 1, 64).expand(3, 96, 64).contiguous()
    scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    fused = mla_q_concat_quant_pad(ql_nope, q_pe, scale, 128)
    ref = _unfused_q(ql_nope, q_pe, scale, 128)
    assert fused.dtype == fp8
    assert torch.equal(fused.view(torch.uint8), ref.view(torch.uint8))


@pytest.mark.parametrize(
    "num_tokens", [1, 2, 7, 63, 64, 65, 128, 193, 513, 1024, 2048, 4096]
)
@pytest.mark.parametrize("num_heads,padded_heads", [(96, 128), (96, 96), (64, 64)])
@pytest.mark.parametrize("scale", [1.0, 0.37])
def test_q_absorb_quant_pad_bitwise(num_tokens, num_heads, padded_heads, scale):
    """The absorbed variant against forward_impl's fp8 MQA path: the W_UK_T bmm
    into the token-major concat buffer, then the concat, quant and pad."""
    torch.manual_seed(num_tokens)
    device = "cuda"
    qk_nope, rope_dim, kv_lora = 128, 64, 512
    q = (torch.randn(num_tokens, num_heads, qk_nope + rope_dim, device=device) * 2).to(
        torch.bfloat16
    )
    # W_UK_T (N, P, L) as the layer holds it: a permuted view of (L, N, P).
    w_uk = (torch.randn(kv_lora, num_heads, qk_nope, device=device) * 0.05).to(
        torch.bfloat16
    )
    w_uk_t = w_uk.permute(1, 2, 0)
    q_nope, q_pe = q.split([qk_nope, rope_dim], dim=-1)
    scale_t = torch.tensor([scale], dtype=torch.float32, device=device)

    cat = q.new_empty((num_tokens, num_heads, kv_lora + rope_dim))
    torch.bmm(q_nope.transpose(0, 1), w_uk_t, out=cat[..., :kv_lora].transpose(0, 1))
    ref = _unfused_q(cat[..., :kv_lora], q_pe, scale_t, padded_heads)
    out = mla_q_absorb_quant_pad(q_nope, q_pe, w_uk_t, scale_t, padded_heads)
    assert out.shape == ref.shape and out.is_contiguous()
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equal, with any NaN matching any NaN."""
    nan = a.isnan()
    if not torch.equal(nan, b.isnan()):
        return False
    bits = torch.int16
    return torch.equal(a.view(bits)[~nan], b.view(bits)[~nan])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sigmoid_gate_mul_every_gate_value(dtype):
    """Every 16-bit gate value, against several attention outputs."""
    device = "cuda"
    gate = torch.arange(-(2**15), 2**15, dtype=torch.int32, device=device)
    gate = gate.to(torch.int16).view(dtype).view(256, 256)
    for x_value in (1.0, -3.0, 0.7, 6.0e4 if dtype == torch.float16 else 3.0e38):
        x = torch.full_like(gate, x_value)
        ref = x * gate.sigmoid()
        out = sigmoid_gate_mul_(x.clone(), gate)
        assert _same_bits(out, ref), x_value


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 12288), (7, 1536), (2048, 12288), (3, 100)])
def test_sigmoid_gate_mul_random(dtype, shape):
    torch.manual_seed(shape[0])
    device = "cuda"
    x = torch.randn(shape, device=device).mul_(4).to(dtype)
    gate = torch.randn(shape, device=device).mul_(8).to(dtype)
    ref = x * gate.sigmoid()
    out = x.clone()
    assert sigmoid_gate_mul_(out, gate) is out
    assert _same_bits(out, ref)


def test_sigmoid_gate_mul_tiny_values():
    """Products in the 16-bit subnormal range keep their bits (no flush)."""
    device = "cuda"
    x = torch.full((4, 4096), 2.0**-126, device=device).to(torch.bfloat16)
    x[1] = 2.0**-130
    x[2] = -(2.0**-133)
    gate = torch.linspace(-30, 30, 4096, device=device).to(torch.bfloat16)
    gate = gate.expand(4, -1).contiguous()
    ref = x * gate.sigmoid()
    assert _same_bits(sigmoid_gate_mul_(x.clone(), gate), ref)
