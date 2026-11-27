# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SiTU-and-mul + MXFP8 quantization of the shared expert intermediate in one kernel (F3 CUDA variant, opt-in K31_F3=1).

Stock (DEP decode, MXFP8 shared expert): vLLM's situ_and_mul_kernel<bf16> writes the BF16 intermediate, then the
down projection's FlashInfer CuTe-DSL MXFP8QuantizeSwizzledKernel (mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True))
reads it back and writes E4M3 values plus 128x4-swizzled UE8M0 scales. This kernel computes the same BF16 values with
the same expression (precise tanhf, the stock operation order, round to BF16) and quantizes them exactly as that
kernel does (block max of the BF16 magnitudes, times fp32(1/448), UE8M0 rounded up, power-of-two inverse scale, clamp
to +-448, cvt.rn.satfinite to E4M3; padding rows of the scale tile zeroed), so both outputs are meant to be bitwise
equal to the stock pair; shared_forward_supported() compares them on a probe batch before use.

The UE8M0, inverse-scale, FP8 conversion and swizzled scale-index helpers follow FlashInfer's
CuTe-DSL quantization utilities (FlashInfer, Apache-2.0).
"""
import functools
import logging

import torch

logger = logging.getLogger(__name__)

_CPP = ("std::vector<torch::Tensor> k31_situ_mxfp8(torch::Tensor gate_up, double beta, double linear_beta);\n")

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // not ATen/cuda/CUDAContext.h: the image has no cusparse.h
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <vector>

// Same expression as vLLM's situ_activation (csrc/activation_kernels.cu)
__device__ __forceinline__ float situ_tanh(float x) { return tanhf(x); }
__device__ __forceinline__ float situ_activation(float g, float u, float beta, float linear_beta, bool clamp_up,
                                                 float inv_beta, float inv_linear_beta) {
    const float gate_out = (0.5f * beta) * situ_tanh(g * inv_beta) * (1.0f + situ_tanh(g * 0.5f));
    const float up_out = clamp_up ? linear_beta * situ_tanh(u * inv_linear_beta) : u;
    return gate_out * up_out;
}

// FlashInfer quantization_cute_dsl_utils.float_to_ue8m0_fast
__device__ __forceinline__ uint32_t float_to_ue8m0(float value) {
    uint32_t r;
    asm("{\n"
        ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
        ".reg .u32 bits, exp_biased, mantissa, bump, result;\n"
        "setp.le.f32 p_zero, %1, 0f00000000;\n"
        "mov.b32 bits, %1;\n"
        "shr.b32 exp_biased, bits, 23;\n"
        "and.b32 exp_biased, exp_biased, 255;\n"
        "and.b32 mantissa, bits, 0x7FFFFF;\n"
        "setp.ne.u32 p_has_mant, mantissa, 0;\n"
        "selp.u32 bump, 1, 0, p_has_mant;\n"
        "setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
        "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
        "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
        "@p_tiny_sub mov.u32 bump, 0;\n"
        "add.u32 result, exp_biased, bump;\n"
        "setp.gt.u32 p_ovf, result, 254;\n"
        "selp.u32 result, 254, result, p_ovf;\n"
        "selp.u32 %0, 0, result, p_zero;\n"
        "}\n" : "=r"(r) : "f"(value));
    return r;
}

// FlashInfer ue8m0_to_inv_scale_fast
__device__ __forceinline__ float ue8m0_to_inv_scale(uint32_t e) {
    float r;
    asm("{\n"
        ".reg .s32 new_exp;\n"
        ".reg .b32 float_bits;\n"
        ".reg .pred p_zero;\n"
        "setp.eq.u32 p_zero, %1, 0;\n"
        "sub.s32 new_exp, 254, %1;\n"
        "max.s32 new_exp, new_exp, 0;\n"
        "shl.b32 float_bits, new_exp, 23;\n"
        "mov.b32 %0, float_bits;\n"
        "@p_zero mov.b32 %0, 0;\n"
        "}\n" : "=f"(r) : "r"(e));
    return r;
}

// FlashInfer bfloat2_to_fp8x2_scaled (two BF16 in one word, low half first)
__device__ __forceinline__ uint32_t bf16x2_to_fp8x2(uint32_t bf2, float inv_scale) {
    uint32_t r;
    asm("{\n"
        ".reg .b32 lo, hi;\n"
        ".reg .f32 f0, f1;\n"
        ".reg .b16 fp8_pair;\n"
        "and.b32 lo, %1, 0xFFFF;\n"
        "shr.b32 hi, %1, 16;\n"
        "shl.b32 lo, lo, 16;\n"
        "shl.b32 hi, hi, 16;\n"
        "mov.b32 f0, lo;\n"
        "mov.b32 f1, hi;\n"
        "mul.f32 f0, f0, %2;\n"
        "mul.f32 f1, f1, %2;\n"
        "min.f32 f0, f0, 0f43E00000;\n"
        "max.f32 f0, f0, 0fC3E00000;\n"
        "min.f32 f1, f1, 0f43E00000;\n"
        "max.f32 f1, f1, 0fC3E00000;\n"
        "cvt.rn.satfinite.e4m3x2.f32 fp8_pair, f1, f0;\n"
        "cvt.u32.u16 %0, fp8_pair;\n"
        "}\n" : "=r"(r) : "r"(bf2), "f"(inv_scale));
    return r;
}

// FlashInfer compute_sf_index_swizzled_128x4_gpu
__device__ __forceinline__ int sf_index_128x4(int row, int col, int padded_cols) {
    return (col % 4) + (col / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (row / 128) * (128 * padded_cols);
}

constexpr int kElts = 8;  // per thread; four threads per 32-element scale block

__global__ void __launch_bounds__(256)
k31_situ_mxfp8_kernel(const __nv_bfloat16* __restrict__ gate_up, uint8_t* __restrict__ out_q,
                      uint8_t* __restrict__ out_sf, const int m, const int padded_m, const int d,
                      const int padded_cols, const float beta, const float linear_beta) {
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long num_threads = static_cast<long>(gridDim.x) * blockDim.x;

    // Padding rows of the last 128-row scale tile: zero scales (as the stock kernel), one 4-column word per store
    const int col_groups = padded_cols / 4;
    for (long j = tid; j < static_cast<long>(padded_m - m) * col_groups; j += num_threads) {
        const int row = m + static_cast<int>(j / col_groups), cg = static_cast<int>(j % col_groups);
        *reinterpret_cast<uint32_t*>(out_sf + sf_index_128x4(row, cg * 4, padded_cols)) = 0u;
    }

    const int units_per_row = d / kElts;
    if (tid >= static_cast<long>(m) * units_per_row)
        return;  // m * units_per_row is a multiple of 32: whole warps return
    const int row = static_cast<int>(tid / units_per_row);
    const int unit = static_cast<int>(tid % units_per_row);
    const bool clamp_up = linear_beta > 0.0f;
    const float inv_beta = 1.0f / beta;
    const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
    const __nv_bfloat16* gate_ptr = gate_up + static_cast<long>(row) * 2 * d + unit * kElts;
    const uint4 g4 = *reinterpret_cast<const uint4*>(gate_ptr);
    const uint4 u4 = *reinterpret_cast<const uint4*>(gate_ptr + d);
    const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(&g4);
    const __nv_bfloat16* u = reinterpret_cast<const __nv_bfloat16*>(&u4);
    __nv_bfloat16 act[kElts];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < kElts; ++ i) {
        act[i] = __float2bfloat16(situ_activation(__bfloat162float(g[i]), __bfloat162float(u[i]), beta, linear_beta,
                                                  clamp_up, inv_beta, inv_linear_beta));
        // |bf16| as fp32 is exact; max over magnitudes as the stock block max
        amax = fmaxf(amax, fabsf(__bfloat162float(act[i])));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float normalized = amax * __uint_as_float(0x3b124925u);  // fp32(1 / 448)
    const uint32_t e = float_to_ue8m0(normalized);
    const float inv_scale = ue8m0_to_inv_scale(e);
    const uint32_t* a2 = reinterpret_cast<const uint32_t*>(act);
    uint32_t w[2];
    #pragma unroll
    for (int i = 0; i < 2; ++ i)
        w[i] = (bf16x2_to_fp8x2(a2[2 * i], inv_scale) & 0xFFFFu) | (bf16x2_to_fp8x2(a2[2 * i + 1], inv_scale) << 16);
    *reinterpret_cast<uint2*>(out_q + static_cast<long>(row) * d + unit * kElts) = make_uint2(w[0], w[1]);
    if (unit % 4 == 0)
        out_sf[sf_index_128x4(row, unit / 4, padded_cols)] = static_cast<uint8_t>(e);
}

std::vector<torch::Tensor> k31_situ_mxfp8(torch::Tensor gate_up, double beta, double linear_beta) {
    TORCH_CHECK(gate_up.dim() == 2 and gate_up.scalar_type() == torch::kBFloat16 and gate_up.is_contiguous());
    const int m = gate_up.size(0), d = gate_up.size(1) / 2;
    TORCH_CHECK(d % 32 == 0 and (d / 32) % 4 == 0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(gate_up.data_ptr()) % 16 == 0);
    const int padded_m = (m + 127) / 128 * 128;
    const int padded_cols = (d / 32 + 3) / 4 * 4;
    auto opts = gate_up.options().dtype(torch::kUInt8);
    auto out_q = torch::empty({m, d}, opts);
    auto out_sf = torch::empty({static_cast<long>(padded_m) * padded_cols}, opts);
    const long threads = static_cast<long>(m) * (d / kElts);  // real rows only (padding scales are spread over them)
    if (m > 0) {
        k31_situ_mxfp8_kernel<<<(threads + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()), out_q.data_ptr<uint8_t>(),
            out_sf.data_ptr<uint8_t>(), m, padded_m, d, padded_cols, static_cast<float>(beta),
            static_cast<float>(linear_beta));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out_q, out_sf};
}
"""


@functools.cache
def _module():
    import os

    from torch.utils.cpp_extension import load_inline

    # Build for this GPU only (cvt.e4m3x2 needs sm_89+; the image's arch list has older targets)
    major, minor = torch.cuda.get_device_capability()
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        return _load(load_inline)
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old


def _load(load_inline):
    return load_inline(
        name="k31_situ_mxfp8_v3",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["k31_situ_mxfp8"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def situ_mxfp8(gate_up: torch.Tensor, beta: float, linear_beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 values [m, d] (float8_e4m3fn) and 128x4-swizzled UE8M0 scales (uint8, padded rows zero) of
    situ_and_mul(gate_up), as mxfp8_e4m3_quantize(situ_and_mul(gate_up), is_sf_swizzled_layout=True)."""
    q, sf = _module().k31_situ_mxfp8(gate_up, float(beta), float(linear_beta))
    return q.view(torch.float8_e4m3fn), sf


def shared_forward_supported(se) -> bool:
    """Whether `shared_forward` serves this shared expert: an unsharded KimiMLP with SiTU whose down projection
    takes a pre-quantized activation, and whose output on a probe batch is bitwise the stock module's."""
    from vllm.model_executor.layers.activation import SituAndMul
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key

    if not (
        isinstance(getattr(se, "act_fn", None), SituAndMul)
        and not getattr(se, "shard_sequence_parallel", False)
        and getattr(se, "gemm_rs_ar", None) is None
        and getattr(se.down_proj, "tp_size", 1) == 1
        and get_input_quant_key(se.down_proj) is not None
    ):
        return False
    try:
        hidden = se.gate_up_proj.input_size
        g = torch.Generator(device="cuda").manual_seed(31)
        x = (torch.randn(60, hidden, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        ok = torch.equal(se(x).view(torch.int16), shared_forward(se, x).view(torch.int16))
    except Exception as e:  # noqa: BLE001
        logger.warning("K31_F3 off: probe failed (%s: %s)", type(e).__name__, str(e)[:200])
        return False
    if not ok:
        logger.warning("K31_F3 off: fused shared expert output differs from the stock module")
    else:
        logger.info("K31_F3 on: fused SiTU + MXFP8 shared expert (probe bitwise equal)")
    return ok


def shared_forward(se, x: torch.Tensor) -> torch.Tensor:
    """KimiMLP.forward for a DEP decode shared expert with the SiTU-and-mul and the down projection's MXFP8
    activation quantization in one kernel (same values as the stock module)."""
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation, get_input_quant_key

    gate_up, _ = se.gate_up_proj(x)
    act = se.act_fn
    linear_beta = -1.0 if act.linear_beta is None else act.linear_beta
    q, sf = situ_mxfp8(gate_up, act.beta, linear_beta)
    qa = QuantizedActivation(data=q, scale=sf, orig_dtype=gate_up.dtype,
                             orig_shape=torch.Size((gate_up.shape[0], gate_up.shape[1] // 2)),
                             quant_key=get_input_quant_key(se.down_proj))
    return se.down_proj.quant_method.apply(se.down_proj, qa)
