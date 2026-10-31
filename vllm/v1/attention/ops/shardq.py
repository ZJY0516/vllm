# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ: Ulysses-style sequence sharding of sparse-MLA prefill under TP.

TP shards Q by heads, so every rank runs the indexer for all tokens and the
sparse-MLA kernel pads its few local heads. ShardQ gives each TP rank a slice of
the query tokens instead: the indexer scores only the rank's tokens, an
all-to-all moves Q from [T, H / tp, D] to [T_rank, H, D], attention runs with
all heads over the replicated KV cache, and a second all-to-all restores
[T, H / tp, D_v]. Each request's tokens fold into ``2 * tp`` pieces; rank ``r``
owns pieces ``r`` and ``2 * tp - 1 - r``, which balances the causal indexer.

Both exchanges push rows straight into peers' symmetric-memory buffers with
one Triton kernel, or fall back to NCCL when the TP group has no symmetric
memory.
"""

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_dcp_group,
    get_tp_group,
    get_world_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.cp_common import DirectCPWorkspace, direct_cp_enabled
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadata,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata


@functools.cache
def shardq_group() -> GroupCoordinator:
    """The TP ranks ShardQ splits tokens over: those holding the same DCP KV
    shard, so the whole TP group without DCP."""
    tp = get_tp_group()
    dcp = get_dcp_group().world_size
    if dcp == 1:
        return tp
    # Without PCP, TP groups are runs of world ranks and DCP groups runs of
    # dcp TP ranks, so a rank shares its KV shard with every dcp-th TP rank.
    world = get_world_group()
    group_ranks = [
        list(range(start + offset, start + tp.world_size, dcp))
        for start in range(0, world.world_size, tp.world_size)
        for offset in range(dcp)
    ]
    return init_model_parallel_group(
        group_ranks,
        world.local_rank,
        dist.get_backend(tp.device_group),
        group_name="shardq",
        use_device_communicator=False,
    )


@dataclass
class ShardQSplit:
    """This step's token split over the ``shardq_group``.

    The split covers the tokens after ``num_decode_tokens`` head-sharded
    decode tokens; its token ids are relative to the first of them.
    """

    world: int
    rank: int
    num_decode_tokens: int
    # Tokens per rank.
    counts: list[int]
    # Every rank's tokens, concatenated in rank order: [num_tokens].
    rank_order: torch.Tensor
    # This rank's tokens (a view of ``rank_order``), ascending.
    local_idx: torch.Tensor
    local_idx_cpu: np.ndarray
    # Destination rank and position there of each ``rank_order`` row.
    row_peer: torch.Tensor
    row_pos: torch.Tensor
    # forward_mqa metadata of this rank's tokens, which run with all heads.
    metadata: "SparseMLACommonMetadata | None" = None

    @property
    def num_local(self) -> int:
        return self.counts[self.rank]


def fold_split(query_start_loc: np.ndarray, world: int) -> list[np.ndarray]:
    """Each rank's token ids under the per-request two-piece fold."""
    pieces = 2 * world
    starts, lens = query_start_loc[:-1], np.diff(query_start_loc)
    bounds = starts[:, None] + (np.arange(pieces + 1) * lens[:, None]) // pieces
    out = []
    for rank in range(world):
        mirror = pieces - 1 - rank
        # Ascending: a request's mirror piece follows its front piece.
        piece_starts = bounds[:, [rank, mirror]].ravel()
        piece_lens = bounds[:, [rank + 1, mirror + 1]].ravel() - piece_starts
        offsets = np.repeat(
            piece_starts - np.cumsum(piece_lens) + piece_lens, piece_lens
        )
        out.append((offsets + np.arange(offsets.size)).astype(np.int64))
    return out


def make_shardq_split(
    query_start_loc: np.ndarray,
    world: int,
    rank: int,
    device: torch.device,
    num_decode_tokens: int = 0,
) -> ShardQSplit:
    per_rank = fold_split(query_start_loc, world)
    counts = [len(x) for x in per_rank]
    host = torch.from_numpy(
        np.stack(
            (
                np.concatenate(per_rank),
                np.repeat(np.arange(world), counts),
                np.concatenate([np.arange(n) for n in counts]),
            )
        )
    )
    if device.type == "cuda":
        host = host.pin_memory()
    rank_order, row_peer, row_pos = host.to(device, non_blocking=True)
    start = sum(counts[:rank])
    return ShardQSplit(
        world=world,
        rank=rank,
        num_decode_tokens=num_decode_tokens,
        counts=counts,
        rank_order=rank_order,
        local_idx=rank_order[start : start + counts[rank]],
        local_idx_cpu=per_rank[rank],
        row_peer=row_peer,
        row_pos=row_pos,
    )


def build_shardq_split(
    common_attn_metadata: "CommonAttentionMetadata",
    vllm_config: "VllmConfig",
    device: torch.device,
) -> ShardQSplit | None:
    """This step's split, or None when too few tokens follow the decodes, none
    of their requests is longer than the top-k, or a rank would own no tokens.

    Decodes are the indexer's: requests of at most 1 + num_speculative_tokens
    tokens. The main MLA layer's builder and its indexer's builder each call
    this and, like every TP rank, derive the same split from the same batch.
    Without a long request, dense MHA or the indexer's causal fill serves the
    prefill with its rows in place.
    """
    spec_config = vllm_config.speculative_config
    num_spec = spec_config.num_speculative_tokens if spec_config else 0
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(common_attn_metadata, decode_threshold=1 + num_spec)
    )
    min_tokens = vllm_config.attention_config.sparse_mla_shard_q_min_tokens
    if num_prefills == 0 or num_prefill_tokens < min_tokens:
        return None
    num_reqs = common_attn_metadata.num_reqs
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound
    assert seq_lens is not None
    topk_tokens = vllm_config.model_config.hf_text_config.index_topk
    if int(seq_lens[num_decodes:num_reqs].max()) <= topk_tokens:
        return None
    qsl = (
        common_attn_metadata.query_start_loc_cpu[num_decodes : num_reqs + 1]
        .numpy()
        .astype(np.int64)
        - num_decode_tokens
    )
    if qsl[-1] != num_prefill_tokens:
        return None
    group = shardq_group()
    split = make_shardq_split(
        qsl, group.world_size, group.rank_in_group, device, num_decode_tokens
    )
    # Attention kernels reject empty batches.
    return split if min(split.counts) > 0 else None


def indexer_rows(split: ShardQSplit) -> tuple[np.ndarray, torch.Tensor]:
    """Rows the indexer scores under ``split``, ascending, on the host and on
    the split's device.

    Every rank scores every decode row and its own split rows, and writes the
    top-k of the p-th scored row to buffer row p: decode rows stay in place,
    and this rank's split rows follow them compactly, in the order the
    token-sharded attention reads them.
    """
    num_decode_tokens = split.num_decode_tokens
    host = np.concatenate(
        (np.arange(num_decode_tokens), split.local_idx_cpu + num_decode_tokens)
    )
    rows = torch.cat(
        (
            torch.arange(num_decode_tokens, device=split.local_idx.device),
            split.local_idx + num_decode_tokens,
        )
    )
    return host, rows


@triton.jit
def _shard_q_kernel(
    q,
    q_stride,
    rank_order,
    row_peer,
    row_pos,
    peer_ptrs,
    dst_stride,
    dst_col,
    width,
    BLOCK: tl.constexpr,
):
    """Push q row ``rank_order[i]`` to rank ``row_peer[i]``, at row
    ``row_pos[i]`` there and column ``dst_col``."""
    i = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < width
    src_row = tl.load(rank_order + i)
    dst_row = tl.load(row_pos + i)
    dst = tl.load(peer_ptrs + tl.load(row_peer + i)).to(
        tl.pointer_type(q.dtype.element_ty)
    )
    vals = tl.load(q + src_row * q_stride + col, mask=mask)
    tl.store(dst + dst_row * dst_stride + dst_col + col, vals, mask=mask)


@triton.jit(do_not_specialize=["num_leading_rows"])
def _unshard_out_kernel(
    out,
    out_stride,
    local_idx,
    peer_ptrs,
    num_leading_rows,
    world,
    width,
    BLOCK: tl.constexpr,
):
    """Push head block ``j`` of local token ``p`` to rank ``j``, at the
    token's row after ``num_leading_rows``."""
    pid = tl.program_id(0)
    p = (pid // world).to(tl.int64)
    j = pid % world
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < width
    dst_row = tl.load(local_idx + p) + num_leading_rows
    dst = tl.load(peer_ptrs + j).to(tl.pointer_type(out.dtype.element_ty))
    vals = tl.load(out + p * out_stride + j * width + col, mask=mask)
    tl.store(dst + dst_row * width + col, vals, mask=mask)


_PUSH_BLOCK = 1024


class ShardQWorkspace(DirectCPWorkspace):
    """The exchange buffers in symmetric memory, with one slot per ubatch."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        num_ubatches: int,
        shapes: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> None:
        super().__init__(group, device, num_ubatches)
        # Each is its storage and every rank's address of each ubatch slot.
        self.buffers = {
            name: self._allocate((num_ubatches, *shape), dtype)
            for name, (shape, dtype) in shapes.items()
        }
        # Any allocation's handle synchronizes the whole group.
        self.handle = self._allocations[0][1]


# Shared by every ShardQ layer; without it the exchanges use NCCL.
_workspace: ShardQWorkspace | None = None


def allocate_shardq_buffers(
    vllm_config: "VllmConfig",
    num_heads: int,
    query_head_dim: int,
    output_head_dim: int,
    query_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    """Allocate the worst-case exchange buffers, if the ``shardq_group`` has
    symmetric memory. ``num_heads`` are a TP rank's."""
    logger.info_once(
        "Sharding sparse MLA prefill steps of >= %d tokens across TP ranks by "
        "token (sparse_mla_shard_q).",
        vllm_config.attention_config.sparse_mla_shard_q_min_tokens,
    )
    global _workspace
    group = shardq_group()
    if _workspace is not None or not direct_cp_enabled(group, torch.bfloat16, None):
        return
    # Under DCP the exchanges move the DCP-gathered heads.
    dcp = get_dcp_group().world_size
    num_heads *= dcp
    max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    # A rank owns at most two pieces of ceil(n / (2 * world)) per request.
    q_rows = (
        -(-max_num_tokens // group.world_size)
        + 2 * vllm_config.scheduler_config.max_num_seqs
    )
    num_ubatches = max(vllm_config.parallel_config.num_ubatches, 1)
    device = torch.device(
        current_platform.device_type, torch.accelerator.current_device_index()
    )
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "q": ((q_rows, group.world_size * num_heads, query_head_dim), query_dtype),
        "out": ((max_num_tokens, num_heads, output_head_dim), output_dtype),
    }
    if dcp > 1:
        # DCP merges the attention outputs by their LSE.
        shapes["lse"] = ((max_num_tokens, num_heads), torch.float32)
    _workspace = ShardQWorkspace(group.device_group, device, num_ubatches, shapes)


def _all_to_all(
    send: torch.Tensor,
    recv: torch.Tensor,
    send_counts: list[int] | None,
    recv_counts: list[int] | None,
) -> None:
    # Exchange bytes: NCCL may lack the dtype (e.g. fp8).
    dist.all_to_all_single(
        recv.view(torch.uint8),
        send.view(torch.uint8),
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        group=shardq_group().device_group,
    )


def shard_q(
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor], split: ShardQSplit
) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, H, D] head shards -> the decode rows, still head-sharded, and this
    rank's token shard with every rank's heads, [T_rank, world * H, D]."""
    if isinstance(q, tuple):
        q = torch.cat(q, dim=-1)
    decode_q, q = q[: split.num_decode_tokens], q[split.num_decode_tokens :]
    _, heads, dim = q.shape
    assert q[0].is_contiguous()
    if _workspace is None:
        # Every rank sends this rank its num_local tokens, in rank order.
        recv = q.new_empty((split.world, split.num_local, heads, dim))
        _all_to_all(q.index_select(0, split.rank_order), recv, split.counts, None)
        return decode_q, torch.cat(list(recv), dim=1)
    storage, peer_ptrs = _workspace.buffers["q"]
    assert storage.shape[2:] == (split.world * heads, dim)
    ubatch = dbo_current_ubatch_id()
    width = heads * dim
    # Peers may still read the previous exchange from their buffers.
    _workspace.handle.barrier(channel=ubatch)
    _shard_q_kernel[(split.rank_order.numel(), triton.cdiv(width, _PUSH_BLOCK))](
        q,
        q.stride(0),
        split.rank_order,
        split.row_peer,
        split.row_pos,
        peer_ptrs[ubatch],
        split.world * width,
        split.rank * width,
        width,
        BLOCK=_PUSH_BLOCK,
    )
    _workspace.handle.barrier(channel=ubatch)
    # Views the buffer, which the next exchange reuses.
    return decode_q, storage[ubatch, : split.num_local]


def _unshard(
    x: torch.Tensor, split: ShardQSplit, decode_x: torch.Tensor | None, name: str
) -> torch.Tensor:
    """[T_rank, world * H, ...] token shard -> [T, H, ...] head shards, after
    the ``decode_x`` rows."""
    assert x[0].is_contiguous()
    row_shape = (x.shape[1] // split.world, *x.shape[2:])
    num_rows = split.num_decode_tokens + sum(split.counts)
    if _workspace is None:
        # One block of heads per destination rank.
        send = torch.stack(x.chunk(split.world, dim=1))
        recv = x.new_empty((sum(split.counts), *row_shape))
        _all_to_all(send, recv, None, split.counts)
        restored = x.new_empty((num_rows, *row_shape))
        restored[split.num_decode_tokens :].index_copy_(0, split.rank_order, recv)
    else:
        storage, peer_ptrs = _workspace.buffers[name]
        assert storage.shape[2:] == row_shape
        ubatch = dbo_current_ubatch_id()
        width = math.prod(row_shape)
        _workspace.handle.barrier(channel=ubatch)
        _unshard_out_kernel[
            (split.num_local * split.world, triton.cdiv(width, _PUSH_BLOCK))
        ](
            x,
            x.stride(0),
            split.local_idx,
            peer_ptrs[ubatch],
            split.num_decode_tokens,
            split.world,
            width,
            BLOCK=_PUSH_BLOCK,
        )
        _workspace.handle.barrier(channel=ubatch)
        # Views the buffer, which the next exchange reuses.
        restored = storage[ubatch, :num_rows]
    if decode_x is not None:
        restored[: split.num_decode_tokens] = decode_x
    return restored


def unshard_out(
    out: torch.Tensor,
    lse: torch.Tensor | None,
    split: ShardQSplit,
    decode_out: torch.Tensor | None,
    decode_lse: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """This rank's token shard of the attention output [T_rank, world * H, D]
    and LSE [T_rank, world * H] -> [T, H, D] and [T, H] head shards, after the
    head-sharded decode rows."""
    out = _unshard(out, split, decode_out, "out")
    if lse is not None:
        lse = _unshard(lse, split, decode_lse, "lse")
    return out, lse
