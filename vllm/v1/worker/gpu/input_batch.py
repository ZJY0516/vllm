# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils import random_uuid
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.attn_utils import FastPrefillBatchMetadata
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.pcp_manager import PCPBatchMetadata
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdmittedDraftCounts,
    )


class InputBuffers:
    def __init__(self, max_num_reqs: int, max_num_tokens: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # DCP: per-request local seq_lens buffer
        self.dcp_local_seq_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )


@dataclass
class InputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding.
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit
    expanded_local_pos: torch.Tensor

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens, (upper bound when using adaptive verification)
    num_scheduled_tokens: np.ndarray
    # number of tokens in the batch,
    #  may be < sum(num_scheduled_tokens) when using adaptive verification
    num_tokens: int
    num_tokens_after_padding: int
    # Sum of draft tokens scheduled across requests.
    num_draft_tokens: int
    # [num_reqs] number of draft tokens scheduled for each request, if any.
    num_draft_tokens_per_req: np.ndarray | None

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs]
    dcp_local_seq_lens: torch.Tensor | None
    # [num_reqs]
    num_computed_tokens_np: np.ndarray
    # [num_reqs]
    prefill_len_np: np.ndarray
    # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray
    # [num_reqs] CPU bool array == (num_computed_prefill_tokens_np < prefill_len_np).
    is_prefilling_np: np.ndarray
    # == np.any(is_prefilling_np)
    has_prefill: bool
    # No prefills, or only prefill rows that run as decodes.
    decode_graph_eligible: bool

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor
    # [num_tokens_after_padding]
    is_padding: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool

    # [num_reqs] per-request prompt length, only populated for R-SWA.
    prompt_lens: torch.Tensor | None

    # [num_reqs] only populated for non-dummy pipeline-parallel batches.
    max_seq_len_np: np.ndarray | None = None

    # Longest query the batch may contain. Set when a cudagraph descriptor promises
    # a query length this batch's own split does not reach, so attention metadata
    # stays valid for every replay the graph serves.
    max_query_len: int | None = None

    # Arms the KV-sharing fast prefill path for this step. Absent for dummy
    # (cudagraph capture) batches, which run the KV-sharing layers in full.
    fast_prefill: "FastPrefillBatchMetadata | None" = None

    # Set only on a PCP rank-local batch.
    pcp: "PCPBatchMetadata | None" = None

    # [num_reqs] prefilling rows that schedule one new prompt token (excluding
    # drafts) over existing context and so compute exactly like decodes.
    # None if there are no prefills.
    prefill_runs_as_decode_np: np.ndarray | None = None

    # [num_reqs] admitted draft counts in flight to the host. Set only under
    # adaptive verification with batch-sharded sampling, when the budget binds.
    num_admitted_draft_tokens: "AdmittedDraftCounts | None" = None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
        max_query_len: int | None = None,
        is_padding: bool = True,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.intp)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # Distribute the remainder evenly so that no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens. Varlen graphs
        # accept any split with non-empty slots, so this shape works for them
        # too; attention metadata is built from the promised max_query_len.
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        assert max_query_len is None or base_tokens + (num_extra > 0) <= max_query_len
        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)
        if num_extra > 0:
            num_scheduled_tokens[-num_extra:] += 1
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens[num_reqs - num_extra : num_reqs] = base_tokens + 1
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        # Pad for full CUDA graph mode.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        input_buffers.is_padding[:num_tokens].fill_(is_padding)
        is_padding = input_buffers.is_padding[:num_tokens]

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Copy so set_dummy_context can add context in place without touching
        # num_scheduled_tokens.
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            has_prefill=False,
            max_seq_len_np=None,
            decode_graph_eligible=True,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=max_query_len,
        )


def set_dummy_context(
    input_batch: InputBatch,
    block_tables: "BlockTables",
    context_len: int,
    num_kv_blocks: int,
    max_model_len: int,
    input_block_tables: Sequence[torch.Tensor] | None = None,
) -> None:
    """Give each dummy request context_len of context, for profiling step cost
    and for DCP FULL cudagraph capture."""
    if input_block_tables is None:
        input_block_tables = block_tables.input_block_tables
    if not input_block_tables:
        # Attention-free models have no KV context to fabricate.
        return
    num_reqs = input_batch.num_reqs
    query_len = input_batch.max_query_len or int(input_batch.num_scheduled_tokens.max())
    context_len = max(min(context_len, max_model_len - query_len), 0)
    if not context_len:
        return

    # Decode-like shape: each request continues after context_len
    # already-computed tokens.
    input_batch.seq_lens += context_len
    input_batch.seq_lens_cpu_upper_bound += context_len
    input_batch.num_computed_tokens_np.fill(context_len)
    input_batch.num_computed_prefill_tokens_np.fill(context_len)
    local_pos = np.arange(input_batch.num_tokens, dtype=np.int64) - np.repeat(
        input_batch.query_start_loc_np[:-1], input_batch.num_scheduled_tokens
    )
    input_batch.positions.copy_(torch.from_numpy(local_pos + context_len))

    seq_len = context_len + query_len
    for block_table, block_size in zip(input_block_tables, block_tables.block_sizes):
        num_blocks = min(cdiv(seq_len, block_size), block_table.shape[1])
        # Spans are disjoint until the pool runs out, then they wrap and share
        # blocks: profiling only needs the reads to be realistic, not distinct.
        block_ids = (
            torch.arange(
                num_reqs * num_blocks,
                dtype=block_table.dtype,
                device=block_table.device,
            )
            % num_kv_blocks
        )
        block_table[:num_reqs, :num_blocks] = block_ids.view(num_reqs, num_blocks)


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    next_prefill_tokens_stride,
    num_lookahead,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
    LOOKAHEAD_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    # Store the next num_lookahead prefill tokens.
    lookahead = tl.arange(0, LOOKAHEAD_BLOCK)
    pos = num_computed + query_len + lookahead
    in_lookahead = lookahead < num_lookahead
    tokens = tl.load(
        request_ptr + pos, mask=in_lookahead & (pos < prefill_len), other=0
    )
    tl.store(
        next_prefill_tokens_ptr
        + lookahead * next_prefill_tokens_stride
        + req_state_idx,
        tokens,
        mask=in_lookahead,
    )


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    num_lookahead = next_prefill_tokens.shape[0]
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        next_prefill_tokens.stride(0),
        num_lookahead,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        BLOCK_SIZE=1024,
        LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead),
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    # NOTE(woosuk): We do +1 because the last thread block is used
    # to pad unused seq_lens as 0 for full CUDA graphs.
    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        # Write the last sampled token ID to input_ids.
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
) -> torch.Tensor:
    assert num_new_sampled_tokens in (0, 1), (
        f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    )
    # use idx_mapping.shape[0] for actual request count
    num_reqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = torch.empty(num_logits, dtype=torch.int64, device=input_ids.device)
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        # NOTE(woosuk): Add num_new_sampled_tokens to ensure the block covers the
        # last sampled token in addition to all draft tokens.
        BLOCK_SIZE=triton.next_power_of_2(
            num_speculative_steps + num_new_sampled_tokens
        ),
    )
    return logits_indices


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    _get_num_sampled_and_rejected_kernel[(num_reqs,)](
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )
    return num_sampled, num_rejected


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    draft_tokens_ptr,
    draft_tokens_out_ptr,
    num_draft_tokens,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    if req_state_idx < 0:
        # Filter rows with negative index entries.
        return

    if draft_tokens_ptr is not None:
        for i in range(num_draft_tokens):
            token_id = tl.load(draft_tokens_ptr + req_id * num_draft_tokens + i)
            tl.store(
                draft_tokens_out_ptr + req_state_idx * num_draft_tokens + i, token_id
            )

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    if query_start_loc_ptr is None:
        query_len = 0
    else:
        query_start = tl.load(query_start_loc_ptr + req_id)
        query_end = tl.load(query_start_loc_ptr + req_id + 1)
        query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    computed_delta = query_len - num_rejected
    if computed_delta != 0:
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + computed_delta)


def post_update(
    # [num_reqs] batch_idx -> req_state_idx; negative index means skip.
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor | None,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor | None,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
    # [num_reqs, num_speculative_steps], contiguous
    draft_tokens: torch.Tensor | None = None,
    # [max_num_reqs, num_speculative_steps], contiguous; required with draft_tokens
    draft_tokens_out: torch.Tensor | None = None,
) -> None:
    num_reqs = idx_mapping.shape[0]
    if draft_tokens is not None:
        # The kernel uses num_speculative_steps as the row stride of both.
        assert draft_tokens_out is not None
        assert draft_tokens.shape[1] == draft_tokens_out.shape[1]
        assert draft_tokens.is_contiguous() and draft_tokens_out.is_contiguous()
    _post_update_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        draft_tokens,
        draft_tokens_out,
        draft_tokens.shape[1] if draft_tokens is not None else 0,
        num_warps=1,
    )


@triton.jit
def _post_update_num_computed_tokens_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


def post_update_num_computed_tokens(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _post_update_num_computed_tokens_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=idx_mapping.device
    )
    _expand_idx_mapping_kernel[(num_reqs,)](
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
    )
    return expanded_idx_mapping, expanded_local_pos


# ---------------------------------------------------------------------------
# K31_NP1: the next-step preparation of a uniform speculative decode step in
# one launch.
#
# A decode step whose requests all verify `Q = 1 + num_spec` tokens prepares
# its inputs and metadata with many small launches and host-to-device copies:
# the runner (request slots, query starts, positions, sequence lengths, input
# ids, logits indices, block tables, slot mappings, the Mamba align
# preprocess, the DCP local lengths) and the attention metadata builders
# (accepted-token gather, aligned state indices, staged KDA metadata, record
# rows, kernel-block tables, indexer slot mapping and decode metadata,
# token-request mapping). Every value is a function of the few host values of
# the batch (request slots and token counts, passed as kernel arguments, so no
# host-to-device copy) and of device state that no launch of that sequence
# changes before reading it. `_np1_step_prep_kernel` writes all of them, with
# the same formulas and the same write sets, in one launch.
#
# The metadata builders still run on the host (they build the Python
# metadata); each launch site that this kernel already served is claimed
# (`np1_claim`) and skipped, so any site the plan does not cover keeps its own
# launch.
# ---------------------------------------------------------------------------

_NP1_PLAN = None


def np1_current():
    """The next-step preparation plan of this step, or None."""
    return _NP1_PLAN


def np1_set(plan) -> None:
    global _NP1_PLAN
    _NP1_PLAN = plan


def np1_claim(key, *params) -> bool:
    """True when this step's plan already wrote the outputs of launch site
    `key` for exactly these parameters (the caller then skips its launch)."""
    plan = _NP1_PLAN
    if plan is None:
        return False
    return plan.claim(key, params)


def np1_observe(key, tensor) -> None:
    """Hand a stock output to this step's plan (comparison mode only)."""
    plan = _NP1_PLAN
    if plan is not None:
        plan.observe(key, tensor)


# Slots of the kernel's tables: `out_tab` / `in_tab` hold int64 data pointers,
# `int_tab` integer parameters.
NP1_O_IS_PADDING = 0  # bool [max_num_tokens]
NP1_O_QSL = 1  # int32 [max_num_reqs + 1]
NP1_O_POSITIONS = 2  # int64 [max_num_tokens]
NP1_O_SEQ_LENS = 3  # int32 [max_num_reqs]
NP1_O_INPUT_IDS = 4  # int32 [max_num_tokens]
NP1_O_SIDX = 5  # int32 [max_num_reqs] Mamba running state column
NP1_O_SRC_COL = 6  # int32 [max_num_reqs]
NP1_O_SRC_OFF = 7  # int32 [max_num_reqs]
NP1_O_NACC = 8  # int32 [max_num_reqs] Mamba accepted tokens
NP1_O_DCP_LOCAL = 9  # int32 [max_num_reqs]
NP1_O_ALIGNED = 10  # int32 [num_mamba_groups, max_num_reqs, slots]
NP1_O_REC_ROWS = 11  # int32 [max_num_reqs]
NP1_O_REQ_PER_TOKEN = 12  # int32 [max_num_batched_tokens]
NP1_O_SLOTS = 13  # int64 [num_kv_cache_groups, max_num_batched_tokens]
NP1_O_KDA = 16  # + 3k: staged state indices, staged query starts, staged accepted
NP1_O_IDX = 32  # + 8k: seq lens, block table, lens, indices, per-request, global, slots
NP1_O_KT = 48  # + m: kernel-block table of attention group m
NP1_O_LEN = 56
NP1_I_NC = 0  # int32 [max_num_reqs] num_computed_tokens
NP1_I_LAST = 1  # int64 [max_num_reqs] last sampled token
NP1_I_DRAFT = 2  # int64 [max_num_reqs, num_spec]
NP1_I_SIDX = 3  # int32 Mamba running state column before the step
NP1_I_NACC = 4  # int32 Mamba accepted tokens before the step
NP1_I_KDA_NACC = 8  # + k: the accepted tokens a KDA builder stages
NP1_I_LEN = 16
NP1_N_DRAFT_STRIDE = 0
NP1_N_ALIGNED_S0 = 1
NP1_N_ALIGNED_S1 = 2
NP1_N_ALIGNED_S2 = 3
NP1_N_MAMBA_STRIDE = 4
NP1_N_SLOT_STRIDE = 5
NP1_N_MAMBA_GID = 8  # + i: kv cache group of Mamba group i
NP1_N_KDA = 16  # + 4k: Mamba group index, staged stride 0, stride 1, slots
NP1_N_IDX = 40  # + 8k: group, out stride, columns, capacity, B, alpha, kernel bs, slots cap
NP1_N_KT = 64  # + 5m: group, B, alpha, out stride, kernel columns
NP1_N_LEN = 88
NP1_MAX_REQS = 4
# K31_S21N8=1 (model_runner): up to eight request slots (kernel constant NR = 8).
NP1_MAX_REQS_WIDE = 8


def np1_arena_layout(q: int, max_reqs: int = NP1_MAX_REQS) -> dict[str, int]:
    """Word offsets (16-byte aligned) of the per-step outputs that the stock
    path allocates, in one int64 arena."""
    t = max_reqs * q
    sizes = (
        ("idx", max_reqs),  # int64 idx_mapping
        ("cu", (max_reqs + 2) // 2),  # int32 cu_num_logits
        ("eidx", t),  # int64 expanded_idx_mapping
        ("logits", t),  # int64 logits_indices
        ("epos", (t + 1) // 2),  # int32 expanded_local_pos
        ("nacc", (max_reqs + 1) // 2),  # int32 gathered accepted tokens
        ("ncomp", (max_reqs + 1) // 2),  # int32 num_computed_tokens
    )
    out, off = {}, 0
    for name, n in sizes:
        out[name] = off
        off += (n + 1) // 2 * 2
    out["len"] = off
    return out


@triton.jit
def _np1_req(r, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR: tl.constexpr):
    """Request slot of batch row `r` (NR = 4 or 8 slots passed as scalars)."""
    if NR == 4:
        return tl.where(r == 0, idx0, tl.where(r == 1, idx1, tl.where(r == 2, idx2, idx3)))
    else:
        lo = tl.where(r == 0, idx0, tl.where(r == 1, idx1, tl.where(r == 2, idx2, idx3)))
        hi = tl.where(r == 4, idx4, tl.where(r == 5, idx5, tl.where(r == 6, idx6, idx7)))
        return tl.where(r < 4, lo, hi)


@triton.jit
def _np1_gathered(
    flat, gid, stride, num_reqs, num_rows, max_rows, idx0, idx1, idx2, idx3, idx4, idx5,
    idx6, idx7, NR: tl.constexpr,
    src_ptrs, dst_ptrs, nb_ptr, nb_stride, mask,
):  # fmt: skip
    """The value `_gather_block_tables_kernel` leaves at flat offset `flat` of
    kv cache group `gid`'s forward block table: a request row's block ids
    below its block count, zero in the cleared padding rows, and elsewhere the
    previous contents (which no launch of the step writes). The loads are
    issued together (the selection follows)."""
    row = flat // stride
    col = flat % stride
    real = row < num_reqs
    req = _np1_req(row, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
    src = tl.load(src_ptrs + gid).to(tl.pointer_type(tl.int32))
    dst = tl.load(dst_ptrs + gid).to(tl.pointer_type(tl.int32))
    nb = tl.load(nb_ptr + gid * nb_stride + req, mask=mask & real, other=0)
    v_src = tl.load(src + req.to(tl.int64) * stride + col, mask=mask & real, other=0)
    v_old = tl.load(dst + flat, mask=mask & (row < max_rows), other=0)
    from_src = real & (col < nb)
    stale = (real & (col >= nb)) | (row >= num_rows)
    return tl.where(from_src, v_src, tl.where(stale, v_old, 0))


@triton.jit
def _np1_kernel_table(
    row, kcol, gid, B, alpha, num_reqs, num_rows, max_rows, idx0, idx1, idx2, idx3,
    idx4, idx5, idx6, idx7, NR: tl.constexpr,
    src_ptrs, dst_ptrs, bt_strides, nb_ptr, nb_stride, mask,
):  # fmt: skip
    """`AttentionGroup.map_to_kernel_block_table` of the forward block table:
    kernel block j of block b is b * alpha + j (B kernel blocks per block)."""
    stride = tl.load(bt_strides + gid)
    v = _np1_gathered(
        row.to(tl.int64) * stride + kcol // B, gid, stride, num_reqs, num_rows,
        max_rows, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR, src_ptrs,
        dst_ptrs, nb_ptr, nb_stride, mask,
    )  # fmt: skip
    return (kcol % B) + alpha * v


@triton.jit
def _np1_slot(
    g, req, pos64, mask, src_bt_ptrs, bt_strides, bt_block_sizes, sm_enabled,
    dcp_sharded, dcp_rank, CP_SIZE: tl.constexpr, CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
):  # fmt: skip
    """`_compute_slot_mappings_kernel` for tokens of kv cache group `g`."""
    bt = tl.load(src_bt_ptrs + g).to(tl.pointer_type(tl.int32))
    bstride = tl.load(bt_strides + g)
    kv_bs = tl.load(bt_block_sizes + g)
    enabled = tl.load(sm_enabled + g)
    is_real = req >= 0
    if CP_SIZE == 1:
        local_pos = pos64
        bidx = tl.where(enabled, local_pos // kv_bs, 0)
        boff = local_pos % kv_bs
        bnum = tl.load(bt + req * bstride + bidx, mask=mask & is_real, other=0)
        sid = bnum * kv_bs + boff
    else:
        sharded = tl.load(dcp_sharded + g)
        vbs = kv_bs * CP_SIZE
        vbi = pos64 // vbs
        vbo = pos64 % vbs
        is_local = vbo // CP_INTERLEAVE % CP_SIZE == dcp_rank
        rounds = vbo // (CP_INTERLEAVE * CP_SIZE)
        remainder = vbo % CP_INTERLEAVE
        local_off = rounds * CP_INTERLEAVE + remainder
        local_pos = vbi * kv_bs + local_off
        local_pos = tl.where(sharded, local_pos, pos64)
        is_local = ~sharded | is_local
        bidx = tl.where(enabled, local_pos // kv_bs, 0)
        boff = local_pos % kv_bs
        bnum = tl.load(
            bt + req * bstride + bidx, mask=mask & is_local & is_real, other=0
        )
        sid = bnum * kv_bs + boff
        sid = tl.where(is_local, sid, PAD_ID)
    return tl.where(enabled & is_real, sid, PAD_ID)


@triton.jit(
    do_not_specialize=[
        "nb_stride", "num_reqs", "num_rows", "idx0", "idx1", "idx2", "idx3",
        "idx4", "idx5", "idx6", "idx7", "max_num_reqs", "max_num_tokens", "dcp_size",
        "dcp_rank", "cp_interleave",
    ]
)  # fmt: skip
def _np1_step_prep_kernel(
    out_tab,
    in_tab,
    int_tab,
    arena_ptr,
    nb_ptr,
    nb_stride,
    prefill_len_ptr,
    src_bt_ptrs,
    dst_bt_ptrs,
    bt_strides,
    bt_block_sizes,
    sm_enabled,
    dcp_sharded,
    num_reqs,
    num_rows,
    idx0,
    idx1,
    idx2,
    idx3,
    idx4,
    idx5,
    idx6,
    idx7,
    max_num_reqs,
    max_num_tokens,
    dcp_size,
    dcp_rank,
    cp_interleave,
    Q: tl.constexpr,
    NR: tl.constexpr,
    BT: tl.constexpr,
    BR: tl.constexpr,
    NG: tl.constexpr,
    NMG: tl.constexpr,
    NSLOT: tl.constexpr,
    NSLOT_P2: tl.constexpr,
    MAMBA_BS: tl.constexpr,
    ALIGN_BS: tl.constexpr,
    NKB: tl.constexpr,
    NIB: tl.constexpr,
    NKT: tl.constexpr,
    IDX_DCP: tl.constexpr,
    IDX_SLOTS: tl.constexpr,
    SKIP_PADDING: tl.constexpr,
    HAS_MAMBA: tl.constexpr,
    HAS_DCP: tl.constexpr,
    HAS_REC_ROWS: tl.constexpr,
    HAS_REQ_PER_TOKEN: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    A_IDX: tl.constexpr,
    A_CU: tl.constexpr,
    A_EIDX: tl.constexpr,
    A_LOGITS: tl.constexpr,
    A_EPOS: tl.constexpr,
    A_NACC: tl.constexpr,
    A_NCOMP: tl.constexpr,
    CHUNK: tl.constexpr,
    NCH: tl.constexpr,
    NCH_IDX: tl.constexpr,
    KT_CH: tl.constexpr,
    EBT_CH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Programs (each a short chain: its loads together, then its stores):
    0 request values, 1 token values, 2 Mamba preprocess, then one per Mamba
    group (aligned indices + the staging of its KDA builders), one per KDA
    builder (staged query starts and accepted tokens), one per kv cache group
    (slot mappings of the tokens), one per indexer builder (decode metadata,
    and its re-derived slot mapping), then the bulk copies and fills."""
    pid = tl.program_id(0)
    num_tokens = num_reqs * Q
    P_MAMBA: tl.constexpr = 2
    P_ALIGN: tl.constexpr = 3
    P_KDAQ: tl.constexpr = P_ALIGN + NMG
    P_SLOT: tl.constexpr = P_KDAQ + NKB
    P_IDX: tl.constexpr = P_SLOT + NG
    P_BULK: tl.constexpr = P_IDX + NIB
    if pid == 0:
        # request values: idx_mapping, cu_num_logits, query_start_loc and
        # sequence lengths (whole buffers), DCP local lengths, the metadata's
        # num_computed_tokens, record rows
        nc_ptr = tl.load(in_tab + 0).to(tl.pointer_type(tl.int32))
        qsl = tl.load(out_tab + 1).to(tl.pointer_type(tl.int32))
        seq_lens = tl.load(out_tab + 3).to(tl.pointer_type(tl.int32))
        rr = tl.arange(0, BR)
        rmask = rr < num_reqs
        req_r = _np1_req(rr, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
        nc_r = tl.load(nc_ptr + req_r, mask=rmask, other=0)
        tl.store(arena_ptr + A_IDX + rr, req_r.to(tl.int64), mask=rmask)
        cu = (arena_ptr + A_CU).to(tl.pointer_type(tl.int32))
        tl.store(cu + rr, rr * Q, mask=rr <= num_reqs)
        ncomp = (arena_ptr + A_NCOMP).to(tl.pointer_type(tl.int32))
        tl.store(ncomp + rr, (nc_r + Q) - Q, mask=rmask)
        if HAS_REC_ROWS:
            rec_rows = tl.load(out_tab + 11).to(tl.pointer_type(tl.int32))
            tl.store(rec_rows + rr, req_r.to(tl.int32), mask=rmask)
        for i0 in tl.range(0, max_num_reqs + 1, BLOCK):
            o = i0 + tl.arange(0, BLOCK)
            tl.store(qsl + o, tl.minimum(o, num_reqs) * Q, mask=o < max_num_reqs + 1)
        for i1 in tl.range(0, max_num_reqs, BLOCK):
            o1 = i1 + tl.arange(0, BLOCK)
            req_o1 = _np1_req(o1, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
            ov = tl.load(nc_ptr + req_o1, mask=o1 < num_reqs, other=0)
            tl.store(
                seq_lens + o1, tl.where(o1 < num_reqs, ov + Q, 0), mask=o1 < max_num_reqs
            )
            if HAS_DCP:
                dcp_out = tl.load(out_tab + 9).to(tl.pointer_type(tl.int32))
                sv = ov + Q
                rounds = sv // (dcp_size * cp_interleave)
                rem = sv % (dcp_size * cp_interleave)
                rem = tl.maximum(rem - dcp_rank * cp_interleave, 0)
                rem = tl.minimum(rem, cp_interleave)
                loc = rounds * cp_interleave + rem
                tl.store(
                    dcp_out + o1, tl.where(o1 < num_reqs, loc, 0), mask=o1 < max_num_reqs
                )
    elif pid == 1:
        # token values: positions, padding flags, expanded indices, logits
        # indices, input ids (last sampled token, then the drafts), token ->
        # request
        nc_ptr = tl.load(in_tab + 0).to(tl.pointer_type(tl.int32))
        last = tl.load(in_tab + 1).to(tl.pointer_type(tl.int64))
        draft = tl.load(in_tab + 2).to(tl.pointer_type(tl.int64))
        d_stride = tl.load(int_tab + 0)
        positions = tl.load(out_tab + 2).to(tl.pointer_type(tl.int64))
        input_ids = tl.load(out_tab + 4).to(tl.pointer_type(tl.int32))
        t = tl.arange(0, BT)
        tmask = t < num_tokens
        r = t // Q
        j = t % Q
        req = _np1_req(r, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
        nc = tl.load(nc_ptr + req, mask=tmask, other=0)
        plen = tl.load(prefill_len_ptr + req, mask=tmask, other=0)
        lv = tl.load(last + req, mask=tmask & (j == 0), other=0)
        dv = tl.load(
            draft + req.to(tl.int64) * d_stride + (j - 1), mask=tmask & (j > 0), other=0
        )
        tl.store(positions + t, nc + j, mask=tmask)
        if SKIP_PADDING:
            is_padding = tl.load(out_tab + 0).to(tl.pointer_type(tl.int8))
            tl.store(is_padding + t, tl.zeros([BT], dtype=tl.int8), mask=tmask)
        tl.store(arena_ptr + A_EIDX + t, req.to(tl.int64), mask=tmask)
        epos = (arena_ptr + A_EPOS).to(tl.pointer_type(tl.int32))
        tl.store(epos + t, j, mask=tmask)
        tl.store(arena_ptr + A_LOGITS + t, (r * Q + j).to(tl.int64), mask=tmask)
        gen = tmask & (nc + Q > plen)
        tl.store(input_ids + t, lv, mask=gen & (j == 0) & (nc >= plen))
        tl.store(input_ids + t, dv, mask=gen & (j > 0))
        if HAS_REQ_PER_TOKEN:
            rpt = tl.load(out_tab + 12).to(tl.pointer_type(tl.int32))
            tl.store(rpt + t, r, mask=tmask)
    elif pid < P_ALIGN:
        if HAS_MAMBA:
            # Mamba align preprocess, then the accepted-token gather
            nc_ptr = tl.load(in_tab + 0).to(tl.pointer_type(tl.int32))
            sidx_in = tl.load(in_tab + 3).to(tl.pointer_type(tl.int32))
            nacc_in = tl.load(in_tab + 4).to(tl.pointer_type(tl.int32))
            sidx_out = tl.load(out_tab + 5).to(tl.pointer_type(tl.int32))
            src_col = tl.load(out_tab + 6).to(tl.pointer_type(tl.int32))
            src_off = tl.load(out_tab + 7).to(tl.pointer_type(tl.int32))
            nacc_out = tl.load(out_tab + 8).to(tl.pointer_type(tl.int32))
            rr = tl.arange(0, BR)
            rmask = rr < num_reqs
            req_r = _np1_req(rr, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
            mmask = rmask & (req_r >= 0)
            nc_r = tl.load(nc_ptr + req_r, mask=rmask, other=0)
            old = tl.load(sidx_in + req_r, mask=mmask, other=-1)
            na = tl.load(nacc_in + req_r, mask=mmask, other=1)
            tl.store(src_col + req_r, old, mask=mmask)
            tl.store(src_off + req_r, tl.maximum(na - 1, 0), mask=mmask)
            new = (nc_r + Q + MAMBA_BS - 1) // MAMBA_BS - 1
            tl.store(sidx_out + req_r, new, mask=mmask)
            reset = (old >= 0) & (old != new)
            tl.store(nacc_out + req_r, tl.full([BR], 1, tl.int32), mask=mmask & reset)
            gathered = (arena_ptr + A_NACC).to(tl.pointer_type(tl.int32))
            tl.store(gathered + rr, tl.where(reset, 1, na), mask=rmask)
    elif pid < P_KDAQ:
        # aligned state indices of Mamba group gi (from where the forward block
        # tables of this launch take them), staged by its KDA builders
        gi = pid - P_ALIGN
        nc_ptr = tl.load(in_tab + 0).to(tl.pointer_type(tl.int32))
        aligned = tl.load(out_tab + 10).to(tl.pointer_type(tl.int32))
        a_s0 = tl.load(int_tab + 1)
        a_s1 = tl.load(int_tab + 2)
        a_s2 = tl.load(int_tab + 3)
        mstride = tl.load(int_tab + 4)
        gid = tl.load(int_tab + 8 + gi)
        rr = tl.arange(0, BR)
        rmask = rr < num_reqs
        req_r = _np1_req(rr, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
        nc_r = tl.load(nc_ptr + req_r, mask=rmask, other=0)
        ss = tl.arange(0, NSLOT_P2)
        first = tl.maximum((nc_r + Q - 1) // ALIGN_BS, 0)
        amask = rmask[:, None] & (ss[None, :] < NSLOT)
        flat = rr[:, None].to(tl.int64) * mstride + first[:, None] + ss[None, :]
        v = _np1_gathered(
            flat, gid, mstride, num_reqs, num_rows, max_num_reqs, idx0, idx1, idx2,
            idx3, idx4, idx5, idx6, idx7, NR, src_bt_ptrs, dst_bt_ptrs, nb_ptr, nb_stride, amask,
        )  # fmt: skip
        tl.store(
            aligned + gi * a_s0 + rr[:, None] * a_s1 + ss[None, :] * a_s2, v, mask=amask
        )
        for kk in tl.static_range(NKB):
            kg = tl.load(int_tab + 16 + 4 * kk)
            st_s0 = tl.load(int_tab + 17 + 4 * kk)
            st_s1 = tl.load(int_tab + 18 + 4 * kk)
            st_n = tl.load(int_tab + 19 + 4 * kk)
            staged = tl.load(out_tab + 16 + 3 * kk).to(tl.pointer_type(tl.int32))
            smask = amask & (kg == gi) & (ss[None, :] < st_n)
            tl.store(staged + rr[:, None] * st_s0 + ss[None, :] * st_s1, v, mask=smask)
    elif pid < P_SLOT:
        # KDA builder k: staged query starts and accepted tokens
        k = pid - P_KDAQ
        sq = tl.load(out_tab + 17 + 3 * k).to(tl.pointer_type(tl.int32))
        sa = tl.load(out_tab + 18 + 3 * k).to(tl.pointer_type(tl.int32))
        src_a = tl.load(in_tab + 8 + k).to(tl.pointer_type(tl.int32))
        rr = tl.arange(0, BR)
        rmask = rr < num_reqs
        av = tl.load(src_a + rr, mask=rmask, other=1)
        tl.store(sq + rr, tl.minimum(rr, num_reqs) * Q, mask=rr <= num_reqs)
        tl.store(sa + rr, av, mask=rmask)
    elif pid < P_IDX:
        # slot mappings of the step's tokens, kv cache group g
        g = pid - P_SLOT
        nc_ptr = tl.load(in_tab + 0).to(tl.pointer_type(tl.int32))
        slots = tl.load(out_tab + 13).to(tl.pointer_type(tl.int64))
        slot_stride = tl.load(int_tab + 5)
        t = tl.arange(0, BT)
        tmask = t < num_tokens
        r = t // Q
        j = t % Q
        req = _np1_req(r, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
        nc = tl.load(nc_ptr + req, mask=tmask, other=0)
        pos64 = (nc + j).to(tl.int64)
        sid = _np1_slot(
            g, req, pos64, tmask, src_bt_ptrs, bt_strides, bt_block_sizes,
            sm_enabled, dcp_sharded, dcp_rank, CP_SIZE, CP_INTERLEAVE, PAD_ID,
        )  # fmt: skip
        tl.store(slots + g * slot_stride + t, sid, mask=tmask)
    elif pid < P_BULK:
        # indexer builder k: decode metadata (per-token bounds, lengths, request
        # indices) and the slot mapping its builder re-derives from its block
        # table
        k = pid - P_IDX
        nc_ptr = tl.load(in_tab + 0).to(tl.pointer_type(tl.int32))
        dsl = tl.load(out_tab + 32 + 8 * k).to(tl.pointer_type(tl.int32))
        dlen = tl.load(out_tab + 34 + 8 * k).to(tl.pointer_type(tl.int32))
        dind = tl.load(out_tab + 35 + 8 * k).to(tl.pointer_type(tl.int32))
        preq = tl.load(out_tab + 36 + 8 * k).to(tl.pointer_type(tl.int32))
        t = tl.arange(0, BT)
        tmask = t < num_tokens
        r = t // Q
        j = t % Q
        req = _np1_req(r, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
        nc = tl.load(nc_ptr + req, mask=tmask, other=0)
        seq = nc + Q
        length = seq - Q + j + 1
        if IDX_DCP:
            glob = tl.load(out_tab + 37 + 8 * k).to(tl.pointer_type(tl.int32))
            tl.store(glob + t, length, mask=tmask)
            base = length // cp_interleave // dcp_size * cp_interleave
            rem2 = length - base * dcp_size
            rem2 = tl.minimum(tl.maximum(rem2 - dcp_rank * cp_interleave, 0), cp_interleave)
            length = base + rem2
        tl.store(dsl + t, length, mask=tmask)
        tl.store(dlen + t, tl.full([BT], 1, tl.int32), mask=tmask)
        tl.store(dind + t, r, mask=tmask)
        rr = tl.arange(0, BR)
        tl.store(preq + rr, tl.full([BR], Q, tl.int32), mask=rr < num_reqs)
        if IDX_SLOTS:
            gid = tl.load(int_tab + 40 + 8 * k)
            kb = tl.load(int_tab + 44 + 8 * k)
            ka = tl.load(int_tab + 45 + 8 * k)
            kbs = tl.load(int_tab + 46 + 8 * k)
            islots = tl.load(out_tab + 38 + 8 * k).to(tl.pointer_type(tl.int64))
            pos64 = (nc + j).to(tl.int64)
            own = _np1_slot(
                gid, req, pos64, tmask, src_bt_ptrs, bt_strides, bt_block_sizes,
                sm_enabled, dcp_sharded, dcp_rank, CP_SIZE, CP_INTERLEAVE, PAD_ID,
            )  # fmt: skip
            cpos = seq - Q + j
            lpos = cpos // (cp_interleave * dcp_size) * cp_interleave + (
                cpos % cp_interleave
            )
            bn = _np1_kernel_table(
                r, lpos // kbs, gid, kb, ka, num_reqs, num_rows, max_num_reqs, idx0,
                idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR,
                src_bt_ptrs, dst_bt_ptrs, bt_strides, nb_ptr,
                nb_stride, tmask,
            )  # fmt: skip
            valid = own != PAD_ID
            sv2 = bn.to(tl.int64) * kbs + (lpos % kbs)
            tl.store(islots + t, tl.where(valid, sv2, PAD_ID), mask=tmask)
    else:
        q0 = pid - P_BULK
        n_bt = NG * num_rows
        q1 = q0 - n_bt
        q2 = q1 - NG * NCH
        n_kt = NKT * num_reqs * KT_CH
        q3 = q2 - n_kt
        n_rows = NIB * num_tokens * EBT_CH
        q4 = q3 - n_rows
        if q0 < n_bt:
            # forward block tables: copy the request rows, clear the others
            g_b1 = q0 // num_rows
            row_b1 = q0 % num_rows
            stride_b1 = tl.load(bt_strides + g_b1)
            dst_b1 = tl.load(dst_bt_ptrs + g_b1).to(tl.pointer_type(tl.int32))
            dst_row_b1 = dst_b1 + row_b1 * stride_b1
            if row_b1 >= num_reqs:
                for iz_b1 in tl.range(0, stride_b1, BLOCK):
                    oz_b1 = iz_b1 + tl.arange(0, BLOCK)
                    tl.store(
                        dst_row_b1 + oz_b1,
                        tl.zeros([BLOCK], tl.int32),
                        mask=oz_b1 < stride_b1,
                    )
            else:
                req_b1 = _np1_req(row_b1, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR)
                src_b1 = tl.load(src_bt_ptrs + g_b1).to(tl.pointer_type(tl.int32))
                nb_b1 = tl.load(nb_ptr + g_b1 * nb_stride + req_b1)
                src_row_b1 = src_b1 + req_b1 * stride_b1
                for i1_b1 in tl.range(0, nb_b1, BLOCK):
                    o1_b1 = i1_b1 + tl.arange(0, BLOCK)
                    v1_b1 = tl.load(src_row_b1 + o1_b1, mask=o1_b1 < nb_b1, other=0)
                    tl.store(dst_row_b1 + o1_b1, v1_b1, mask=o1_b1 < nb_b1)
        elif q1 < NG * NCH:
            # slot mappings: PAD from the step's token count to the end
            g_b2 = q1 // NCH
            c_b2 = q1 % NCH
            slots_b2 = tl.load(out_tab + 13).to(tl.pointer_type(tl.int64))
            slot_stride_b2 = tl.load(int_tab + 5)
            lo_b2 = tl.maximum(c_b2 * CHUNK, num_tokens)
            hi_b2 = tl.minimum((c_b2 + 1) * CHUNK, max_num_tokens)
            for i2_b2 in tl.range(lo_b2, hi_b2, BLOCK):
                o2_b2 = i2_b2 + tl.arange(0, BLOCK)
                tl.store(
                    slots_b2 + g_b2 * slot_stride_b2 + o2_b2,
                    tl.full([BLOCK], PAD_ID, tl.int64),
                    mask=o2_b2 < hi_b2,
                )
        elif q2 < n_kt:
            # kernel-block tables of the request rows, BLOCK columns per program
            c_b3 = q2 % KT_CH
            m_b3 = q2 // KT_CH // num_reqs
            row_b3 = q2 // KT_CH % num_reqs
            gid_b3 = tl.load(int_tab + 64 + 5 * m_b3)
            kb_b3 = tl.load(int_tab + 65 + 5 * m_b3)
            ka_b3 = tl.load(int_tab + 66 + 5 * m_b3)
            ostride_b3 = tl.load(int_tab + 67 + 5 * m_b3)
            kcols_b3 = tl.load(int_tab + 68 + 5 * m_b3)
            kt_b3 = tl.load(out_tab + 48 + m_b3).to(tl.pointer_type(tl.int32))
            o3_b3 = c_b3 * BLOCK + tl.arange(0, BLOCK)
            mk_b3 = o3_b3 < kcols_b3
            v3_b3 = _np1_kernel_table(
                row_b3, o3_b3, gid_b3, kb_b3, ka_b3, num_reqs, num_rows,
                max_num_reqs, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR,
                src_bt_ptrs, dst_bt_ptrs,
                bt_strides, nb_ptr, nb_stride, mk_b3,
            )  # fmt: skip
            tl.store(kt_b3 + row_b3.to(tl.int64) * ostride_b3 + o3_b3, v3_b3, mask=mk_b3)
        elif q3 < n_rows:
            # indexer: each token's copy of its request's (kernel) block-table
            # row, BLOCK columns per program
            c_b4 = q3 % EBT_CH
            k_b4 = q3 // EBT_CH // num_tokens
            tok_b4 = q3 // EBT_CH % num_tokens
            row_b4 = tok_b4 // Q
            gid_b4 = tl.load(int_tab + 40 + 8 * k_b4)
            ostride_b4 = tl.load(int_tab + 41 + 8 * k_b4)
            cols_b4 = tl.load(int_tab + 42 + 8 * k_b4)
            kb_b4 = tl.load(int_tab + 44 + 8 * k_b4)
            ka_b4 = tl.load(int_tab + 45 + 8 * k_b4)
            ebt_b4 = tl.load(out_tab + 33 + 8 * k_b4).to(tl.pointer_type(tl.int32))
            o4_b4 = c_b4 * BLOCK + tl.arange(0, BLOCK)
            mk_b4 = o4_b4 < cols_b4
            v4_b4 = _np1_kernel_table(
                row_b4, o4_b4, gid_b4, kb_b4, ka_b4, num_reqs, num_rows,
                max_num_reqs, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, NR,
                src_bt_ptrs, dst_bt_ptrs,
                bt_strides, nb_ptr, nb_stride, mk_b4,
            )  # fmt: skip
            tl.store(ebt_b4 + tok_b4.to(tl.int64) * ostride_b4 + o4_b4, v4_b4, mask=mk_b4)
        elif q4 < NIB * NCH_IDX:
            # indexer: zero the per-token bounds past the step's tokens, and
            # PAD its slot mapping there
            k_b5 = q4 // NCH_IDX
            c_b5 = q4 % NCH_IDX
            cap_b5 = tl.load(int_tab + 43 + 8 * k_b5)
            dsl_b5 = tl.load(out_tab + 32 + 8 * k_b5).to(tl.pointer_type(tl.int32))
            lo_b5 = tl.maximum(c_b5 * CHUNK, num_tokens)
            hi_b5 = tl.minimum((c_b5 + 1) * CHUNK, cap_b5)
            for i5_b5 in tl.range(lo_b5, hi_b5, BLOCK):
                o5_b5 = i5_b5 + tl.arange(0, BLOCK)
                tl.store(dsl_b5 + o5_b5, tl.zeros([BLOCK], tl.int32), mask=o5_b5 < hi_b5)
            if IDX_SLOTS:
                scap_b5 = tl.load(int_tab + 47 + 8 * k_b5)
                islots_b5 = tl.load(out_tab + 38 + 8 * k_b5).to(tl.pointer_type(tl.int64))
                hi6_b5 = tl.minimum((c_b5 + 1) * CHUNK, scap_b5)
                for i6_b5 in tl.range(lo_b5, hi6_b5, BLOCK):
                    o6_b5 = i6_b5 + tl.arange(0, BLOCK)
                    tl.store(
                        islots_b5 + o6_b5, tl.full([BLOCK], -1, tl.int64), mask=o6_b5 < hi6_b5
                    )


def np1_grid(
    ng: int, nmg: int, nkb: int, nib: int, nkt: int, nch: int, nch_idx: int,
    num_reqs: int, num_rows: int, q: int, kt_ch: int, ebt_ch: int,
) -> tuple[int]:  # fmt: skip
    """The launch grid of `_np1_step_prep_kernel`."""
    t = num_reqs * q
    return (
        3 + nmg + nkb + ng + nib + ng * num_rows + ng * nch
        + nkt * num_reqs * kt_ch + nib * t * ebt_ch + nib * nch_idx,
    )  # fmt: skip
