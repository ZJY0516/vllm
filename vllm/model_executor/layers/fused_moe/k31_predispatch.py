# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-dispatch (M1, opt-in K31_M1=1): the dispatch head of a DeepGEMM NVFP4 MegaMoE launch as its own early kernel.

Right after the router top-k (and the route stage of the input staging, which writes the mapped ids into the symmetric
buffer), one CTA on a side stream does what the head of the MegaMoE kernel does before its token pulls:
  - assigns every (token, top-k) of this rank a slot in its expert's list from this rank and writes the source index
    into the destination rank's workspace,
  - writes this rank's per-expert counts into every destination's [source rank][local expert] table,
  - passes the before-pull NVLink barrier (the same barrier and sequence slot the kernel would use),
  - publishes the receive totals with the arrival tag the kernel's scheduler waits for,
  - bumps the epoch and leaves `kK31PreBit | epoch` for the MegaMoE kernel of this launch (header patched by
    patch_m1.py), which then starts at the token pulls.
The kernel overlaps the latent down projection and the shared expert. It changes which slot a token takes inside its
expert's list (as any order of the stock kernel's block atomics may), not what any token computes.
"""
import functools
import os

import torch

_HEADER_MARK = "kK31PreWordIdx"

_CPP = "void k31_predispatch(torch::Tensor topk_idx, int64_t num_tokens, std::vector<int64_t> peers, int64_t rank);\n"

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // not ATen/cuda/CUDAContext.h: the image has no cusparse.h
#include <c10/cuda/CUDAException.h>
#include <vector>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

using namespace deep_gemm;

// Same spare words as the patched MegaMoE header (patch_m1.py)
constexpr uint32_t kK31PreWordIdx = layout::MegaMoESignals::kNumMaxExperts - 1;
constexpr uint32_t kK31EpochWordIdx = layout::MegaMoESignals::kNumMaxExperts - 2;
constexpr uint32_t kK31XReadyBaseIdx = layout::MegaMoESignals::kNumMaxExperts - 128;
constexpr uint64_t kK31PreBit = 1ull << 62;

// Enumerators (prvalues): the layout constructors take `const uint32_t&`, which would odr-use host constexprs
enum : uint32_t {
    kNumRanks = @RANKS@, kNumExperts = @EXPERTS@, kNumTopk = @TOPK@,
    kNumMaxTokensPerRank = @MAXT@, kNumMainSMs = @SMS@, kNumThreads = 512,
    kNumExpertsPerRank = kNumExperts / kNumRanks,
    kBeforeDispatchPullBarrierTag = 1,
};
static_assert(kNumExperts <= kK31XReadyBaseIdx and kNumRanks <= 32, "M1 spare words");

__global__ void __launch_bounds__(kNumThreads, 1)
k31_mega_predispatch_kernel(const int64_t* topk_idx, const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer) {
#if defined(__CUDA_ARCH__)  // SymBuffer's map/get_base_ptr exist only in the device pass
    const auto workspace = layout::Workspace(sym_buffer.get_base_ptr(), kNumRanks, kNumExperts,
                                             kNumMaxTokensPerRank, kNumTopk, 0);
    const uint32_t thread_idx = threadIdx.x;
    __shared__ uint32_t count[kNumExperts];
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads)
        count[i] = 0;
    __syncthreads();

    // Slots and source indices: this CTA claims all of this rank's slots, so each list is 0 .. n - 1
    for (uint32_t i = thread_idx; i < num_tokens * kNumTopk; i += kNumThreads) {
        const int expert_idx = static_cast<int>(topk_idx[i]);
        if (expert_idx >= 0) {
            const auto dst_slot_idx = atomicAdd_block(count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, expert_idx / kNumExpertsPerRank) = i;
        }
    }
    __syncthreads();

    // This rank's counts into every destination's [source rank][local expert] table
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads)
        *sym_buffer.map(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, i % kNumExpertsPerRank),
                        i / kNumExpertsPerRank) = static_cast<uint64_t>(count[i]);
    __syncthreads();

    // The before-pull NVLink barrier of this launch (the MegaMoE kernel skips it when pre-dispatched)
    comm::nvlink_barrier<kNumRanks, 1, kNumThreads, 0, kBeforeDispatchPullBarrierTag>(
        workspace, sym_buffer, 0, thread_idx, []() { __syncthreads(); }, false, false);
    __syncthreads();

    // Receive totals with the arrival tag of the MegaMoE kernel's scheduler (compact dispatch)
    for (uint32_t i = thread_idx; i < kNumExpertsPerRank; i += kNumThreads) {
        uint64_t total = 0;
        #pragma unroll
        for (uint32_t j = 0; j < kNumRanks; ++ j)
            total += *workspace.get_expert_recv_count_ptr(j, i);
        ptx::st_rel(workspace.get_expert_recv_count_sum_ptr(i),
                    (static_cast<uint64_t>(kNumMainSMs * kNumRanks) << 32) | total);
    }

    // Epoch of this launch and the pre-dispatched word for the MegaMoE kernel
    if (thread_idx == 0) {
        auto* words = workspace.signals->expert_send_count;
        const uint64_t epoch = words[kK31EpochWordIdx] + 1;
        words[kK31EpochWordIdx] = epoch;
        words[kK31PreWordIdx] = kK31PreBit | epoch;
    }
#endif
}

void k31_predispatch(torch::Tensor topk_idx, int64_t num_tokens, std::vector<int64_t> peers, int64_t rank) {
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64 and topk_idx.is_contiguous());
    TORCH_CHECK(static_cast<uint32_t>(peers.size()) == kNumRanks);
    const layout::SymBuffer<kNumRanks> sym_buffer(peers, static_cast<uint32_t>(rank));
    k31_mega_predispatch_kernel<<<1, kNumThreads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        topk_idx.data_ptr<int64_t>(), static_cast<uint32_t>(num_tokens), sym_buffer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_DG = None


def set_deep_gemm(module) -> None:
    """Use this DeepGEMM package (benches that load one by path) instead of vLLM's."""
    global _DG
    _DG = module


def _dg():
    if _DG is not None:
        return _DG
    from vllm.utils.deep_gemm import _import_deep_gemm

    return _import_deep_gemm()


def _include_dir() -> str:
    return os.path.join(os.path.dirname(_dg().__file__), "include")


def header_supports_m1() -> bool:
    """Whether the loaded DeepGEMM package has the patched (pre-dispatch aware) NVFP4 MegaMoE header."""
    path = os.path.join(_include_dir(), "deep_gemm", "impls", "sm100_nvfp4_mega_moe.cuh")
    try:
        with open(path) as f:
            return _HEADER_MARK in f.read()
    except OSError:
        return False


@functools.cache
def _module(ranks: int, experts: int, topk: int, maxt: int, sms: int):
    from torch.utils.cpp_extension import load_inline

    src = (_CUDA.replace("@RANKS@", str(ranks)).replace("@EXPERTS@", str(experts)).replace("@TOPK@", str(topk))
           .replace("@MAXT@", str(maxt)).replace("@SMS@", str(sms)))
    inc = _include_dir()
    # Build for this GPU only (the image's arch list includes targets older than some PTX the headers carry)
    major, minor = torch.cuda.get_device_capability()
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        return _load(src, inc, ranks, experts, topk, maxt, sms)
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old


def _load(src, inc, ranks, experts, topk, maxt, sms):
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name=f"k31_mega_predispatch3_r{ranks}_e{experts}_k{topk}_t{maxt}_s{sms}",
        cpp_sources=_CPP,
        cuda_sources=src,
        functions=["k31_predispatch"],
        extra_include_paths=[inc],
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
        extra_cflags=["-O2", "-std=c++20"],
        verbose=False,
    )


def predispatch(symm_buffer, num_tokens: int) -> None:
    """Launch the early dispatch kernel on the current stream for this rank's `num_tokens` staged top-k ids."""
    dg = _dg()
    ranks = symm_buffer.group.size()
    mod = _module(ranks, symm_buffer.num_experts, symm_buffer.num_topk, symm_buffer.num_max_tokens_per_rank,
                  dg.get_num_sms())
    mod.k31_predispatch(symm_buffer.topk_idx, num_tokens, list(symm_buffer.handle.buffer_ptrs),
                        symm_buffer.group.rank())
