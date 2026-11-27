# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-row MXFP8 activation quantization (opt-in, K31_MXFP8_FASTQ=1): one kernel with the arithmetic of FlashInfer's CuTe-DSL
MXFP8QuantizeSwizzledKernel (block max of the BF16 magnitudes, times fp32(1/448), UE8M0 rounded up, power-of-two
inverse scale, clamp to +-448, cvt.rn.satfinite to E4M3, 128x4-swizzled scales with the padding rows zeroed), laid
out for decode rows: eight values per thread, four threads per 32-value block, a grid over the real rows. The
outputs are meant to be bitwise mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)'s; the MXFP8 linear kernel (kernels/linear/mxfp8/flashinfer.py) checks
that on its first call per width and falls back otherwise.

The UE8M0 rounding helper follows FlashInfer's quantization_cute_dsl_utils.float_to_ue8m0_fast
(FlashInfer, Apache-2.0).
"""
import functools
import os

import torch

_CPP = "std::vector<torch::Tensor> k31_mxfp8_quant(torch::Tensor x);\n"

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // not ATen/cuda/CUDAContext.h: the image has no cusparse.h
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <vector>

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

constexpr int kElts = 8;  // per thread; four threads per 32-value scale block

__global__ void __launch_bounds__(256)
k31_mxfp8_quant_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf,
                       const int m, const int padded_m, const int k, const int padded_cols) {
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long num_threads = static_cast<long>(gridDim.x) * blockDim.x;
    const int col_groups = padded_cols / 4;
    for (long j = tid; j < static_cast<long>(padded_m - m) * col_groups; j += num_threads) {
        const int row = m + static_cast<int>(j / col_groups), cg = static_cast<int>(j % col_groups);
        *reinterpret_cast<uint32_t*>(out_sf + sf_index_128x4(row, cg * 4, padded_cols)) = 0u;
    }
    const int units_per_row = k / kElts;
    if (tid >= static_cast<long>(m) * units_per_row)
        return;  // m * units_per_row is a multiple of 32: whole warps return
    const int row = static_cast<int>(tid / units_per_row);
    const int unit = static_cast<int>(tid % units_per_row);
    const uint4 v4 = *reinterpret_cast<const uint4*>(x + static_cast<long>(row) * k + unit * kElts);
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&v4);
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < kElts; ++ i)
        amax = fmaxf(amax, fabsf(__bfloat162float(v[i])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float normalized = amax * __uint_as_float(0x3b124925u);  // fp32(1 / 448)
    const uint32_t e = float_to_ue8m0(normalized);
    const float inv_scale = ue8m0_to_inv_scale(e);
    const uint32_t* a2 = reinterpret_cast<const uint32_t*>(&v4);
    uint32_t w[2];
    #pragma unroll
    for (int i = 0; i < 2; ++ i)
        w[i] = (bf16x2_to_fp8x2(a2[2 * i], inv_scale) & 0xFFFFu) | (bf16x2_to_fp8x2(a2[2 * i + 1], inv_scale) << 16);
    *reinterpret_cast<uint2*>(out_q + static_cast<long>(row) * k + unit * kElts) = make_uint2(w[0], w[1]);
    if (unit % 4 == 0)
        out_sf[sf_index_128x4(row, unit / 4, padded_cols)] = static_cast<uint8_t>(e);
}

std::vector<torch::Tensor> k31_mxfp8_quant(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2 and x.scalar_type() == torch::kBFloat16 and x.is_contiguous());
    const int m = x.size(0), k = x.size(1);
    TORCH_CHECK(k % 128 == 0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
    const int padded_m = (m + 127) / 128 * 128;
    const int padded_cols = (k / 32 + 3) / 4 * 4;
    auto opts = x.options().dtype(torch::kUInt8);
    auto out_q = torch::empty({m, k}, opts);
    auto out_sf = torch::empty({static_cast<long>(padded_m) * padded_cols}, opts);
    const long threads = static_cast<long>(m) * (k / kElts);
    if (m > 0) {
        k31_mxfp8_quant_kernel<<<(threads + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out_q.data_ptr<uint8_t>(), out_sf.data_ptr<uint8_t>(),
            m, padded_m, k, padded_cols);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out_q, out_sf};
}
"""


@functools.cache
def _module():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"  # cvt.e4m3x2 needs sm_89+
    try:
        return load_inline(name="k31_mxfp8q_v1", cpp_sources=_CPP, cuda_sources=_CUDA,
                           functions=["k31_mxfp8_quant"], extra_cuda_cflags=["-O3"], verbose=False)
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old


def mxfp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E4M3 [m, k] float8_e4m3fn, 128x4-swizzled UE8M0 scales uint8) of a contiguous BF16 [m, k]."""
    q, sf = _module().k31_mxfp8_quant(x)
    return q.view(torch.float8_e4m3fn), sf
