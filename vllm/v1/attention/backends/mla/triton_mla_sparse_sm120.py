# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse-MLA fallback backend for NoPE (rope-free) models on SM120.

Why this exists
---------------
GLM-5.3-Flash (``glm5_next``) uses NoPE MLA (``qk_rope_head_dim == 0``, query
dim 512) with a DeepSeek-style top-k indexer whose effective buffer width is
2176 (``index_topk=2048`` widened by ``index_kpool=4`` and rounded up to a
multiple of 128). On SM120 (RTX PRO 6000 / consumer Blackwell) neither shipped
CUDA sparse-MLA kernel covers that combination:

  * FLASHINFER_MLA_SPARSE_SM120 -> the released FlashInfer dispatch tables cover
    NoPE/d_qk=512 only up to top-k 1024 (DSV4 table); the top-k 2048 slot lives
    only in the d_qk=576 (rope) DSV3_2 table. So GLM's (NoPE + ~2176) pair is
    not dispatchable, and it additionally requires the packed fp8_ds_mla cache.
  * FLASHMLA / FLASHATTN sparse -> SM90/SM100 only.

The clean fix is upstream FlashInfer PR #4842 (adds a GLM53_NOPE kernel with the
native 2176 top-k). Until that lands in a FlashInfer release AND vLLM bumps its
pinned FlashInfer version, there is no in-tree way to serve this model on SM120.

This backend is that bridge. It reuses ALL of the in-tree sparse-MLA plumbing
(the metadata builder, the indexer, the ``flat_kv_row_view`` /
``triton_convert_req_index_to_global_index`` index conversion, and the latent
value up-projection in the MLA wrapper) and adds only the attention compute,
delegating it to a portable split-KV Triton sparse-MLA kernel
(``ops/triton_mla_sparse_kernel.py``). That kernel derives its geometry from the
query head dim at dispatch, so it serves both the pure-NoPE 512 layout
(glm5_next) and the 576 rope layout, and it masks the ``-1`` / out-of-range top-k
tail internally, so any top-k width works with no per-width instantiation.

Scope and removal
-----------------
Deliberately narrow: SM120 only, NoPE only (``qk_rope_head_dim == 0``),
``glm5_next`` model family only, bf16 latent KV cache only. It never shadows a
shipped CUDA kernel: on any model / dtype / capability outside that niche
``supports_combination`` returns a reason string and the selector moves on. The
cuda.py priority list places ``FLASHINFER_MLA_SPARSE_SM120`` ahead of this
backend, and the NoPE branch of the Tier-1 guard in
``flashinfer_mla_sparse.py::FlashInferMLASparseSM120Backend.supports_combination``
makes FlashInfer decline the NoPE case so selection falls through to here.

REMOVE this backend once FlashInfer #4842 is in a released FlashInfer that vLLM
pins in ``requirements/cuda.txt`` and ``FlashInferMLASparseSM120Backend`` serves
``glm5_next`` end-to-end. At that point the NoPE branch of the Tier-1 guard also
comes out.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _FlashInferMLASparseBackendBase,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# NoPE MLA latent width (kv_lora_rank). The rope-free query and the KV latent are
# both this wide; d_qk == d_v == 512.
_NOPE_LATENT_DIM = 512


class TritonMLASparseSM120Backend(_FlashInferMLASparseBackendBase):
    """SM120 sparse-MLA backend backed by the portable split-KV Triton kernel."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # bf16 latent cache only. The fp8_ds_mla packed layout would need a
    # gather+dequant front-end (see PR #49026's fp8 path) and is deferred; this
    # fallback is correctness-first.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_SM120"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64, 256]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return TritonMLASparseSM120Impl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config

        if not use_sparse:
            return "TRITON_MLA_SPARSE_SM120 only serves sparse (indexer) MLA"
        if dtype != torch.bfloat16:
            return "TRITON_MLA_SPARSE_SM120 requires bfloat16 activations"
        if kv_cache_dtype not in (None, "auto", "bfloat16"):
            return (
                "TRITON_MLA_SPARSE_SM120 requires a bf16 latent KV cache "
                f"(auto/bfloat16); got kv_cache_dtype={kv_cache_dtype!r}"
            )

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            model_type = getattr(hf_text_config, "model_type", "") or ""
            if not model_type.startswith("glm5_next"):
                return (
                    "TRITON_MLA_SPARSE_SM120 is scoped to the glm5_next model "
                    f"family; got model_type={model_type!r}"
                )
            if getattr(hf_text_config, "index_topk", None) is None:
                return (
                    "TRITON_MLA_SPARSE_SM120 requires a model with an "
                    "index_topk (indexer-sparse) config"
                )
            # This fallback exists precisely for the NoPE case that the shipped
            # FlashInfer SM120 kernel cannot dispatch. With-rope models must use
            # FLASHINFER_MLA_SPARSE_SM120.
            if getattr(hf_text_config, "qk_rope_head_dim", None) != 0:
                return (
                    "TRITON_MLA_SPARSE_SM120 only serves NoPE MLA "
                    "(qk_rope_head_dim == 0); use FLASHINFER_MLA_SPARSE_SM120 "
                    "for with-rope models"
                )
        return None


class TritonMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """Rope-free (NoPE) sparse-MLA decode on SM120 via the split-KV Triton kernel.

    Mirrors ``FlashInferMLASparseSM120Impl`` one-to-one: it performs the sparse
    latent attention in the ``kv_lora_rank`` space and returns the latent output
    ``[num_tokens, num_heads, kv_lora_rank]``; the value up-projection (W_UV) is
    absorbed into ``o_proj`` by the MLA wrapper, exactly as for the FlashInfer
    path. The single difference is the attention kernel: the portable split-KV
    Triton kernel (``triton_mla_sparse_attention``) instead of the FlashInfer
    TRTLLM-gen launcher. That kernel does its own online softmax and masks the
    ``-1`` / out-of-range top-k tail, so no separate valid-length bound, chunked
    scratch, or attention-sink finalization is needed here.
    """

    is_sparse = True
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "TRITON_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "TRITON_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "TRITON_MLA_SPARSE_SM120 requires a bf16 latent KV cache; got "
                f"kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        if self.qk_rope_head_dim != 0:
            raise NotImplementedError(
                "TRITON_MLA_SPARSE_SM120 only supports NoPE MLA "
                f"(qk_rope_head_dim == 0); got {self.qk_rope_head_dim}."
            )
        if self.kv_lora_rank != _NOPE_LATENT_DIM:
            raise NotImplementedError(
                "TRITON_MLA_SPARSE_SM120's kernel is specialized for "
                f"kv_lora_rank == {_NOPE_LATENT_DIM}; got {self.kv_lora_rank}."
            )

        # Skip-topk layers are built with indexer=None and receive the shared
        # buffer via mla_args instead (cf. the FlashInfer SM120 impl).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        assert self.topk_indices_buffer is not None

        # Cache the device SM count for the kernel's split-KV heuristic, and
        # pre-compile the autotune configs it may pick so the first real request
        # doesn't pay the inline JIT / config-sweep cost.
        self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_triton_sparse()

    def _warmup_triton_sparse(self) -> None:
        """Prime the kernel's ``@triton.autotune`` caches at init across the
        split-count candidates so the first decode step doesn't stall on JIT."""
        buffer = self.topk_indices_buffer
        assert buffer is not None
        device = buffer.device
        topk = buffer.shape[-1]
        dim_qk = self.kv_lora_rank  # 512 for NoPE
        q = torch.empty(1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device)
        kv = torch.empty(64, 1, dim_qk, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # Flat [row, latent] view of the paged bf16 KV cache; global indices
        # address into it. row width == kv_lora_rank (512) for NoPE.
        kv_flat, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )

        # Convert request-local indexer top-k to global cache rows. With
        # return_valid_counts the valid entries are compacted to a contiguous
        # prefix and the tail is guaranteed -1, which the Triton kernel masks
        # out (indices >= 0 & < seq_kv), so the per-token valid count is unused.
        topk_indices_physical, _seq_lens = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        # Kernel contract: q [tokens, heads, dim_qk], kv [rows, 1, dim_qk],
        # indices [tokens, 1, topk]. Returns latent output [tokens, heads, 512].
        output = triton_mla_sparse_attention(
            q,
            kv_flat.unsqueeze(1),
            topk_indices_physical.unsqueeze(1),
            sm_scale=self.scale,
            sm_count=self._sm_count,
        )
        return output, None
