# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.ops.causal_conv1d_metadata import (
    CausalConv1dMetadata,
    compute_causal_conv1d_metadata,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.stateful_attn import (
    BaseStatefulAttentionMetadata,
    BaseStatefulAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

M = TypeVar("M", bound="BaseMambaAttentionMetadata")


@dataclass
class BaseMambaAttentionMetadata(BaseStatefulAttentionMetadata):
    # cu_chunk_seqlen_p is a tensor of shape (nchunks+1,) that contains, for
    # each chunk, its offsets into the varlen sequence dimension. It is defined
    # such that the i-th chunk contains tokens from cu_chunk_seqlen_p[i] to
    # cu_chunk_seqlen_p[i+1].
    cu_chunk_seqlen_p: torch.Tensor | None = None
    # last_chunk_indices_p is a tensor of shape (batch,) that contains the
    # index of the last chunk for every sequence in the (prefill) batch.
    last_chunk_indices_p: torch.Tensor | None = None

    causal_conv1d: CausalConv1dMetadata | None = None
    # ReplaySSM standard decode — Triton: per-row ring cursor and flush flag,
    # plus (decode_rows, ngroups, replayssm_buffer_len) fp32 scratch for
    # precomputed B·C products. All None when use_replayssm is disabled or
    # the FlashInfer ReplaySSM backend is selected.
    write_pos_d: torch.Tensor | None = None
    is_flush_d: torch.Tensor | None = None
    bc_pre_scratch: torch.Tensor | None = None
    # ReplaySSM — FlashInfer checkpointing_ssu two-kernel scratch.
    replayssm_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


class BaseMambaAttentionMetadataBuilder(BaseStatefulAttentionMetadataBuilder[M]):
    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.use_replayssm = vllm_config.cache_config.use_replayssm
        self.replayssm_buffer_len = vllm_config.cache_config.replayssm_buffer_len
        self.use_flashinfer_replayssm = (
            self.use_replayssm
            and vllm_config.mamba_config.backend == MambaBackendEnum.FLASHINFER
        )

        scheduler_config = vllm_config.scheduler_config
        self.decode_bc_pre_scratch: torch.Tensor | None = None
        self.decode_replayssm_scratch: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        # ReplaySSM CUDA-graph buffers for the selected backend.
        if self.use_replayssm and not self.use_flashinfer_replayssm:
            self.decode_write_pos_d: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.decode_is_flush_d: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int8,
                device=device,
            )
            # B_cache shape = (ngroups, replayssm_buffer_len, dstate); the page
            # layout is (conv_state, ssm_state, x_cache, dt_cache, B_cache).
            bc_ngroups = kv_cache_spec.shapes[4][0]
            bc_scratch_bs = max(
                self.decode_cudagraph_max_bs, scheduler_config.max_num_seqs
            )
            self.decode_bc_pre_scratch = torch.empty(
                (
                    bc_scratch_bs,
                    bc_ngroups,
                    self.replayssm_buffer_len,
                ),
                dtype=torch.float32,
                device=device,
            )
        elif self.use_flashinfer_replayssm:
            from flashinfer.mamba.checkpointing_ssu import (
                allocate_checkpointing_ssu_scratch,
            )

            nheads = kv_cache_spec.shapes[2][0]
            self.decode_replayssm_scratch = allocate_checkpointing_ssu_scratch(
                batch_size=scheduler_config.max_num_seqs,
                num_heads=nheads,
                num_predicted_tokens=1,
                max_window=self.replayssm_buffer_len,
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        prev_last_scheduled_idx: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> M:
        """Default build implementation for Mamba-like attention backends.
        Subclasses (e.g., Mamba2) can override to add additional metadata.
        """
        metadata = self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            prev_last_scheduled_idx=prev_last_scheduled_idx,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )

        if metadata.num_prefills > 0:
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-metadata.num_prefills - 1 :]
                - metadata.num_decode_tokens
            )
            metadata.causal_conv1d = compute_causal_conv1d_metadata(
                query_start_loc_p_cpu,
                device=common_attn_metadata.query_start_loc.device,
            )
        if self.use_replayssm:
            self._build_replayssm_metadata(metadata, common_attn_metadata)
        return self._update_metadata_for_cudagraph_capture(metadata)

    def _compute_chunk_metadata(
        self,
        chunk_size: int,
        num_prefills: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        """Compute chunk-specific metadata for Mamba models.

        The code below carefully constructs the chunks such that:
        1. Chunks contain tokens from a *single* sequence only.
        2. For every sequence, we are guaranteed that we can
           retrieve the mamba state *every* chunk_size tokens.
        Constraint (1) dramatically simplifies the mamba kernels.
        Constraint (2) dramatically simplifies the implementation
        of prefix caching for mamba (wip). We need to take care
        of the interaction with chunked prefill in order to
        satisfy constraint (2).
        """
        # TODO (tdoublep): This code could probably be optimized.
        cu_chunk_seqlen = []
        seq_idx = []
        last_chunk_indices = []
        seqlen_pos = 0

        for req_idx in range(num_prefills):
            this_num_computed = num_computed_tokens_p_cpu[req_idx].item()
            this_new_tokens = (
                query_start_loc_p_cpu[req_idx + 1].item()
                - query_start_loc_p_cpu[req_idx].item()
            )

            # if computed tokens are not chunk-aligned, use the first
            # chunk to finish it off
            if this_num_computed % chunk_size != 0:
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                # how many tokens to finish the chunk?
                chunk_len = (
                    cdiv(this_num_computed, chunk_size) * chunk_size - this_num_computed
                )
                # we can only use at most this_new_tokens
                chunk_len = min(chunk_len, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            n_chunks = cdiv(this_new_tokens, chunk_size)
            for chunk in range(n_chunks):
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                chunk_len = min(chunk_size, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            assert this_new_tokens == 0
            last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

        cu_chunk_seqlen.append(seqlen_pos)

        return cu_chunk_seqlen, seq_idx, last_chunk_indices

    def _prefill_cpu_metadata(
        self,
        common: M,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill context lengths and query offsets, from CPU data only.

        `seq_lens_cpu_upper_bound` is precise for prefill rows in all modes
        (including async spec decode), so this avoids the D2H sync that
        `compute_num_computed_tokens().cpu()` would force.

        Returns (num_computed_tokens_p_cpu, query_start_loc_p_cpu).
        """
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        num_reqs = common.num_reqs
        num_prefills = common.num_prefills
        query_start_loc_p_cpu = (
            common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
            - common.num_decode_tokens
        )
        prefill_query_lens_cpu = query_start_loc_p_cpu[1:] - query_start_loc_p_cpu[:-1]
        num_computed_tokens_p_cpu = (
            seq_lens_cpu[num_reqs - num_prefills : num_reqs] - prefill_query_lens_cpu
        )
        return num_computed_tokens_p_cpu, query_start_loc_p_cpu

    def _build_chunk_metadata_tensors(
        self,
        chunk_size: int,
        common: M,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute chunk metadata and return as device tensors.
        Returns (cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p).
        """
        num_prefills = common.num_prefills

        num_computed_tokens_p_cpu, query_start_loc_p_cpu = self._prefill_cpu_metadata(
            common, common_attn_metadata
        )

        cu_chunk_seqlen, seq_idx, last_chunk_indices = self._compute_chunk_metadata(
            chunk_size,
            num_prefills,
            num_computed_tokens_p_cpu,
            query_start_loc_p_cpu,
        )

        device = common_attn_metadata.query_start_loc.device
        # Build on pinned CPU and upload non-blocking to avoid the synchronous
        # H2D copy that `torch.as_tensor(list, device=cuda)` would force.
        cu_chunk_seqlen_p = async_tensor_h2d(
            cu_chunk_seqlen, dtype=torch.int32, device=device
        )
        seq_idx_p = async_tensor_h2d(seq_idx, dtype=torch.int32, device=device)
        last_chunk_indices_p = async_tensor_h2d(
            last_chunk_indices, dtype=torch.int32, device=device
        )
        return cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p

    def _build_replayssm_metadata(
        self, metadata: M, common_attn_metadata: CommonAttentionMetadata
    ) -> None:
        num_decodes = metadata.num_decodes
        write_pos_d = None
        is_flush_d = None
        replayssm_scratch = None
        if self.use_replayssm and not self.use_flashinfer_replayssm and num_decodes > 0:
            decode_base_cpu = common_attn_metadata.replayssm_decode_base_cpu
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            async_spec_decode = (
                self.vllm_config.scheduler_config.async_scheduling
                and self.vllm_config.speculative_config is not None
            )
            if decode_base_cpu is None or seq_lens_cpu is None or async_spec_decode:
                raise ValueError(
                    "--use-replayssm requires exact CPU sequence lengths and "
                    "decode-base counts to derive decode write positions"
                )
            query_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1 : num_decodes + 1]
                - common_attn_metadata.query_start_loc_cpu[:num_decodes]
            )
            num_computed_d = seq_lens_cpu[:num_decodes] - query_lens_cpu
            decode_base_d = decode_base_cpu[:num_decodes]
            align_mode = self.vllm_config.cache_config.mamba_cache_mode == "align"
            block_size = self.kv_cache_spec.block_size
            if align_mode:
                # After a boundary the align copy leaves an exact checkpoint at
                # the block start and the new block's ring restarts empty, so
                # re-anchor there; max() keeps the prompt-end anchor for the
                # first (partial) block.
                effective_base = torch.maximum(
                    decode_base_d, (num_computed_d // block_size) * block_size
                )
            else:
                effective_base = decode_base_d
            # write_pos counts decode steps since the ring's last full-state
            # write (the anchor), so a resumed request re-anchors correctly.
            decode_steps_cpu = num_computed_d - effective_base
            valid_decode_rows = query_lens_cpu > 0
            # A single-token prefill row replayed as decode (query_len==1 with
            # prior state) has decode_steps < 0; force it to a one-token flush
            # (write_pos=0, is_flush=1). The flush branch reads an empty history
            # window, so it applies exactly one recurrence step off the checkpoint
            # -- identical to the baseline decode kernel for that row. The split
            # (treat_short_extends_as_decodes=False) admits only such rows here.
            leftover_prompt = valid_decode_rows & (decode_steps_cpu < 0)
            decode_steps_cpu = torch.where(
                valid_decode_rows & ~leftover_prompt,
                decode_steps_cpu,
                torch.zeros_like(decode_steps_cpu),
            )
            write_pos_cpu = torch.remainder(decode_steps_cpu, self.replayssm_buffer_len)
            is_flush_cpu = (
                write_pos_cpu == self.replayssm_buffer_len - 1
            ) | leftover_prompt
            if align_mode:
                # Force a flush on the step completing a mamba block so the exact
                # boundary state is materialized for prefix caching.
                is_flush_cpu = is_flush_cpu | (
                    valid_decode_rows
                    & ((num_computed_d + query_lens_cpu) % block_size == 0)
                )
            is_flush_cpu = is_flush_cpu.to(torch.int8)
            write_pos_d = async_tensor_h2d(
                write_pos_cpu.to(torch.int32).tolist(),
                dtype=torch.int32,
                device=common_attn_metadata.query_start_loc.device,
            )
            is_flush_d = async_tensor_h2d(
                is_flush_cpu.tolist(),
                dtype=torch.int8,
                device=common_attn_metadata.query_start_loc.device,
            )

        if self.use_flashinfer_replayssm and num_decodes > 0:
            assert self.decode_replayssm_scratch is not None
            cb_scaled, cumAdt_vec, cb_old = self.decode_replayssm_scratch
            replayssm_scratch = (
                cb_scaled[:num_decodes],
                cumAdt_vec[:num_decodes],
                cb_old[:num_decodes],
            )

        bc_pre_scratch = None
        if (
            self.use_replayssm
            and self.decode_bc_pre_scratch is not None
            and num_decodes > 0
        ):
            bc_pre_scratch = self.decode_bc_pre_scratch[:num_decodes]

        metadata.write_pos_d = write_pos_d
        metadata.is_flush_d = is_flush_d
        metadata.bc_pre_scratch = bc_pre_scratch
        metadata.replayssm_scratch = replayssm_scratch

    def _update_metadata_for_cudagraph_capture(self, metadata: M) -> M:
        metadata = super()._update_metadata_for_cudagraph_capture(metadata)
        if not self.use_replayssm:
            return metadata
        write_pos_d = metadata.write_pos_d
        is_flush_d = metadata.is_flush_d
        bc_pre_scratch = metadata.bc_pre_scratch
        replayssm_scratch = metadata.replayssm_scratch
        if (
            metadata.num_prefills == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            padded_bs = metadata.num_reqs
            if self.use_replayssm and not self.use_flashinfer_replayssm:
                assert write_pos_d is not None
                assert is_flush_d is not None
                self.decode_write_pos_d[: metadata.num_decodes].copy_(
                    write_pos_d[: metadata.num_decodes],
                    non_blocking=True,
                )
                write_pos_d = self.decode_write_pos_d[:padded_bs]
                write_pos_d[metadata.num_decodes :] = 0

                self.decode_is_flush_d[: metadata.num_decodes].copy_(
                    is_flush_d[: metadata.num_decodes],
                    non_blocking=True,
                )
                is_flush_d = self.decode_is_flush_d[:padded_bs]
                is_flush_d[metadata.num_decodes :] = 0

                if self.decode_bc_pre_scratch is not None:
                    bc_pre_scratch = self.decode_bc_pre_scratch[:padded_bs]
            elif self.use_flashinfer_replayssm:
                assert self.decode_replayssm_scratch is not None
                cb_scaled, cumAdt_vec, cb_old = self.decode_replayssm_scratch
                replayssm_scratch = (
                    cb_scaled[:padded_bs],
                    cumAdt_vec[:padded_bs],
                    cb_old[:padded_bs],
                )

        return replace(
            metadata,
            write_pos_d=write_pos_d,
            is_flush_d=is_flush_d,
            bc_pre_scratch=bc_pre_scratch,
            replayssm_scratch=replayssm_scratch,
        )
