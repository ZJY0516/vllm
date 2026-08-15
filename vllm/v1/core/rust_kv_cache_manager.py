# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python orchestration for the self-contained Rust KV cache manager."""

from collections.abc import Iterable
from typing import Any

from vllm.distributed.kv_events import KVCacheEvent
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

try:
    from vllm._rust_kv_cache import FullAttentionKVCacheManager as _NativeManager
except ImportError as exc:  # pragma: no cover - depends on the build configuration.
    raise ImportError(
        "VLLM_USE_RUST_KV_CACHE_MANAGER=1 requires the vllm._rust_kv_cache extension"
    ) from exc


class _CoordinatorFacade:
    enable_partial_hash_hits = False

    def __init__(self, manager: "RustFullAttentionKVCacheManager") -> None:
        self._manager = manager

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Iterable[KVCacheBlock], ...],
        **_: Any,
    ) -> int:
        (blocks,) = new_computed_blocks
        return self._manager._core.get_num_blocks_to_allocate(
            request_id, num_tokens, [block.block_id for block in blocks]
        )


class _BlockPoolFacade:
    def __init__(self, manager: "RustFullAttentionKVCacheManager") -> None:
        self._manager = manager

    def get_num_free_blocks(self) -> int:
        return self._manager._core.num_free_blocks

    def get_usage(self) -> float:
        return self._manager._core.usage

    def free_blocks(self, blocks: Iterable[KVCacheBlock]) -> None:
        del blocks
        raise RuntimeError(
            "deferred block release is not supported by the Rust KV cache manager"
        )


class RustFullAttentionKVCacheManager:
    """Drop-in manager for one full-attention KV cache group.

    Rust owns all mutable cache metadata. Python validates the supported
    configuration, converts scheduler request data into batched native calls,
    and wraps returned block IDs for the existing scheduler interface.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        num_prefill_lookahead: int = 0,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        watermark: float = 0.0,
    ) -> None:
        del max_in_flight_tokens
        self._validate_config(
            kv_cache_config=kv_cache_config,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            use_eagle=use_eagle,
            num_prefill_lookahead=num_prefill_lookahead,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.block_size = scheduler_block_size
        self.enable_caching = enable_caching
        self.log_stats = log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None
        if watermark < 0:
            raise ValueError("watermark must be non-negative")
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        self._core = _NativeManager(
            kv_cache_config.num_blocks, self.block_size, enable_caching
        )
        self._blocks = tuple(
            KVCacheBlock(block_id) for block_id in range(kv_cache_config.num_blocks)
        )
        self.empty_kv_cache_blocks = KVCacheBlocks(((),))
        self.coordinator = _CoordinatorFacade(self)
        self.block_pool = _BlockPoolFacade(self)

    @staticmethod
    def _validate_config(
        *,
        kv_cache_config: KVCacheConfig,
        scheduler_block_size: int,
        hash_block_size: int,
        use_eagle: bool,
        num_prefill_lookahead: int,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        metrics_collector: KVCacheMetricsCollector | None,
    ) -> None:
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError(
                "The Rust KV cache manager currently requires one KV cache group."
            )
        spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if not isinstance(spec, FullAttentionSpec):
            raise ValueError(
                "The Rust KV cache manager currently supports full attention only."
            )
        if not (
            spec.block_size == scheduler_block_size == hash_block_size
            and dcp_world_size == 1
            and pcp_world_size == 1
        ):
            raise ValueError(
                "The Rust KV cache manager requires identical cache, scheduler, "
                "and hash block sizes with DCP=PCP=1."
            )
        unsupported = []
        if use_eagle or num_prefill_lookahead:
            unsupported.append("EAGLE/MTP")
        if enable_kv_cache_events:
            unsupported.append("KV cache events")
        if metrics_collector is not None:
            unsupported.append("KV cache metrics")
        if kv_cache_config.needs_kv_cache_zeroing:
            unsupported.append("KV cache zeroing")
        if unsupported:
            raise ValueError(
                "The Rust KV cache manager does not support: " + ", ".join(unsupported)
            )

    def _wrap_block_ids(self, block_ids: Iterable[int]) -> KVCacheBlocks:
        ids = list(block_ids)
        if not ids:
            return self.empty_kv_cache_blocks
        return KVCacheBlocks((tuple(self._blocks[block_id] for block_id in ids),))

    @property
    def usage(self) -> float:
        return self._core.usage

    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        return self.enable_caching and not request.skip_reading_prefix_cache

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0
        block_ids, num_computed_tokens = self._core.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return self._wrap_block_ids(block_ids), num_computed_tokens, 0

    def get_computed_blocks_for_connector(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        raise RuntimeError("KV connectors are not supported by the Rust manager")

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")
        if num_external_computed_tokens or num_encoder_tokens:
            raise ValueError(
                "external and encoder KV allocation is not supported by the "
                "Rust manager"
            )
        computed = new_computed_blocks or self.empty_kv_cache_blocks
        (computed_group,) = computed.blocks
        computed_ids = [block.block_id for block in computed_group]
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(num_local_computed_tokens, self.max_model_len)
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )
        num_tokens_to_cache = 0
        if self.enable_caching and not delay_cache_blocks:
            num_tokens_to_cache = min(num_tokens_main_model, request.num_tokens)
        watermark_blocks = 0
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks
        full_num_tokens = (
            min(request.num_tokens, self.max_model_len)
            if full_sequence_must_fit
            else None
        )
        new_block_ids = self._core.allocate_slots(
            request.request_id,
            num_tokens_need_slot,
            computed_ids,
            request.block_hashes,
            num_tokens_to_cache,
            reserved_blocks,
            watermark_blocks,
            full_num_tokens,
        )
        return None if new_block_ids is None else self._wrap_block_ids(new_block_ids)

    def free(self, request: Request) -> None:
        self._core.free(request.request_id)

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        del request
        raise RuntimeError("deferred frees are not supported by the Rust manager")

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        del request_id, processed_computed_tokens, num_prompt_tokens

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        return self._wrap_block_ids(self._core.get_block_ids(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return (self._core.get_block_ids(request_id),)

    def get_block_ids_for_computed_tokens(
        self, request_id: str, num_computed_tokens: int
    ) -> tuple[list[int], ...]:
        block_ids = self._core.get_block_ids(request_id)
        return (block_ids[: cdiv(num_computed_tokens, self.block_size)],)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [self._core.get_num_common_prefix_blocks(running_request_id)]

    def estimate_cached_tokens(self, request: Request) -> int:
        return self._core.estimate_cached_tokens(request.request_id)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        if self.enable_caching:
            self._core.cache_blocks(
                request.request_id, request.block_hashes, num_computed_tokens
            )

    def evict_blocks(self, block_ids: set[int]) -> None:
        self._core.evict_blocks(list(block_ids))

    def reset_prefix_cache(self) -> bool:
        reset = self._core.reset_prefix_cache()
        if reset and self.prefix_cache_stats is not None:
            self.prefix_cache_stats.reset = True
        return reset

    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:
        if not self.log_stats or not self.prefix_cache_lookup_enabled(request):
            return
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
        )

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def truncate_computed_blocks(
        self, blocks: KVCacheBlocks, num_computed_tokens: int
    ) -> KVCacheBlocks:
        (group,) = blocks.blocks
        return KVCacheBlocks((group[: num_computed_tokens // self.block_size],))

    def take_events(self) -> list[KVCacheEvent]:
        return []

    def take_new_block_ids(self) -> list[int]:
        return []

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        return [], []

    def take_partial_tail_offloads(self) -> dict[str, list[tuple[int, int, int]]]:
        return {}

    def get_zeroing_block_ids_in_range(
        self, request_id: str, start_token: int, end_token: int
    ) -> list[int]:
        del request_id, start_token, end_token
        return []

    def record_blocks_for_zeroing(self, request_id: str, start_token: int) -> None:
        del request_id, start_token
        raise RuntimeError("KV cache zeroing is not supported by the Rust manager")

    def new_step_starts(self) -> None:
        pass
