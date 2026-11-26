# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec


def align_mamba_prefill_checkpoints(
    checkpoints: Mapping[int, Mapping[str, tuple[int, int]]] | None,
    req_ids: Sequence[str],
) -> dict[int, tuple[list[int], list[int]]] | None:
    """Align scheduler-supplied prefill checkpoints with batch row order."""
    if not checkpoints:
        return None
    return {
        group_id: (
            [
                group_ckpts[req_id][0] if req_id in group_ckpts else 0
                for req_id in req_ids
            ],
            [
                group_ckpts[req_id][1] if req_id in group_ckpts else NULL_BLOCK_ID
                for req_id in req_ids
            ],
        )
        for group_id, group_ckpts in checkpoints.items()
    }


@dataclass
class MambaPrefillCheckpointMetadata:
    checkpoint_offsets: torch.Tensor
    state_indices: torch.Tensor
    # Host copies of `checkpoint_offsets` and of each row's query length, for
    # backends that lay out the checkpoint before any tensor exists (Mamba2's
    # SSD chunk split, FlashInfer GDN's checkpoint plan).
    offsets: list[int] | None = None
    query_lens: list[int] | None = None


class MambaPrefillCheckpointBuilder:
    """Build backend-neutral prefill checkpoint locations from scheduler plan."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: MambaSpec,
        checkpoint_alignment: int | None,
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_spec = kv_cache_spec
        self.checkpoint_alignment = checkpoint_alignment

    def build(
        self,
        m: CommonAttentionMetadata,
        request_rows: list[int],
    ) -> MambaPrefillCheckpointMetadata | None:
        if (
            not self.vllm_config.cache_config.enable_prefix_caching
            or self.checkpoint_alignment is None
            or m.mamba_prefill_checkpoints is None
        ):
            return None
        all_offsets, all_block_ids = m.mamba_prefill_checkpoints
        checkpoint_offsets = [
            all_offsets[row] if row < len(all_offsets) else 0 for row in request_rows
        ]
        if not any(checkpoint_offsets):
            return None
        state_indices = [
            all_block_ids[row]
            if row < len(all_block_ids) and offset > 0
            else NULL_BLOCK_ID
            for row, offset in zip(request_rows, checkpoint_offsets)
        ]
        all_query_lens = m.query_start_loc_cpu.diff().tolist()
        query_lens = [all_query_lens[row] for row in request_rows]
        device = m.query_start_loc.device
        return MambaPrefillCheckpointMetadata(
            async_tensor_h2d(checkpoint_offsets, dtype=torch.int32, device=device),
            async_tensor_h2d(state_indices, dtype=torch.int32, device=device),
            offsets=checkpoint_offsets,
            query_lens=query_lens,
        )


@dataclass(frozen=True)
class MambaPrefillCheckpointExporter:
    """Store each checkpoint's pre-convolution window and recurrent state.

    The recurrent states are rows of a buffer the prefill kernel emitted;
    `recurrent_rows` picks each checkpoint row's state out of it, and by
    default row `i` holds checkpoint `i`'s. Layers with only a conv state
    pass no recurrent buffers. `token_indices` maps the rows' tokens, laid out
    by `cu_seqlens`, to their rows of `raw_qkv`.
    """

    state_len: int | None = None

    def export(
        self,
        checkpoint: MambaPrefillCheckpointMetadata,
        *,
        raw_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        recurrent_checkpoint: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        recurrent_rows: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
    ) -> None:
        state_len = (
            self.state_len if self.state_len is not None else conv_state.shape[-1]
        )
        # The block is cached as a checkpoint, so it needs its whole conv window.
        assert checkpoint.offsets is None or all(
            offset == 0 or offset >= state_len for offset in checkpoint.offsets
        )
        assert (
            recurrent_rows is None
            or recurrent_rows.numel() == checkpoint.checkpoint_offsets.numel()
        )
        width = raw_qkv.shape[-1]
        has_recurrent = recurrent_checkpoint is not None
        if recurrent_checkpoint is None:
            # Unread placeholders for the recurrent pointers.
            recurrent_checkpoint = recurrent_state = conv_state
            recurrent_row_size = 0
        else:
            assert recurrent_state is not None
            recurrent_row_size = recurrent_checkpoint[0].numel()
        block_size = 512
        store_cache_checkpoints_kernel[
            (
                checkpoint.checkpoint_offsets.numel(),
                triton.cdiv(max(width * state_len, recurrent_row_size), block_size),
            )
        ](
            raw_qkv,
            conv_state,
            recurrent_checkpoint,
            recurrent_state,
            cu_seqlens,
            checkpoint.checkpoint_offsets,
            checkpoint.state_indices,
            recurrent_rows if recurrent_rows is not None else checkpoint.state_indices,
            token_indices if token_indices is not None else cu_seqlens,
            raw_qkv.stride(0),
            raw_qkv.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            recurrent_checkpoint.stride(0),
            recurrent_state.stride(0),
            checkpoint.checkpoint_offsets.stride(0),
            state_len,
            width,
            recurrent_row_size,
            NULL_BLOCK_ID,
            has_recurrent,
            recurrent_rows is not None,
            token_indices is not None,
            block_size,
        )


@triton.jit
def store_cache_checkpoints_kernel(
    x_ptr,
    conv_state_ptr,
    recurrent_checkpoint_ptr,
    recurrent_state_ptr,
    query_start_loc_ptr,
    checkpoint_offsets_ptr,
    checkpoint_state_indices_ptr,
    recurrent_rows_ptr,
    token_indices_ptr,
    x_stride_0,
    x_stride_1,
    state_stride_0,
    state_stride_1,
    state_stride_2,
    checkpoint_stride_0,
    recurrent_state_stride_0,
    checkpoint_offset_stride,
    STATE_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    RECURRENT_ROW_SIZE: tl.constexpr,
    NULL_STATE_IDX: tl.constexpr,
    HAS_RECURRENT: tl.constexpr,
    HAS_RECURRENT_ROWS: tl.constexpr,
    HAS_TOKEN_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx).to(tl.int64)
    checkpoint_offset = tl.load(
        checkpoint_offsets_ptr + seq_idx * checkpoint_offset_stride
    )
    valid_checkpoint = (state_idx != NULL_STATE_IDX) & (checkpoint_offset > 0)
    # Only the leading programs cover conv columns; the rest skip the window.
    if tl.program_id(1) * BLOCK_SIZE < WIDTH * STATE_LEN:
        valid_conv = (
            (cols < WIDTH * STATE_LEN)
            & valid_checkpoint
            & (checkpoint_offset >= STATE_LEN)
        )
        width_idx = cols // STATE_LEN
        history_idx = cols % STATE_LEN
        checkpoint_end = tl.load(query_start_loc_ptr + seq_idx) + checkpoint_offset
        token_idx = checkpoint_end - STATE_LEN + history_idx
        if HAS_TOKEN_MAP:
            token_idx = tl.load(token_indices_ptr + token_idx, mask=valid_conv, other=0)
        values = tl.load(
            x_ptr + token_idx.to(tl.int64) * x_stride_0 + width_idx * x_stride_1,
            mask=valid_conv,
        )
        tl.store(
            conv_state_ptr
            + state_idx * state_stride_0
            + width_idx * state_stride_1
            + history_idx * state_stride_2,
            values,
            mask=valid_conv,
        )

    if HAS_RECURRENT:
        if HAS_RECURRENT_ROWS:
            recurrent_row = tl.load(recurrent_rows_ptr + seq_idx).to(tl.int64)
        else:
            recurrent_row = seq_idx.to(tl.int64)
        valid_recurrent = (cols < RECURRENT_ROW_SIZE) & valid_checkpoint
        recurrent = tl.load(
            recurrent_checkpoint_ptr + recurrent_row * checkpoint_stride_0 + cols,
            mask=valid_recurrent,
        )
        tl.store(
            recurrent_state_ptr + state_idx * recurrent_state_stride_0 + cols,
            recurrent,
            mask=valid_recurrent,
        )
