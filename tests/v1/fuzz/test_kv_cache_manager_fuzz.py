# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hypothesis state-machine fuzz for the v1 KVCacheManager.

Drives get_computed_blocks / allocate_slots / free / cache_blocks /
reset_prefix_cache directly (no scheduler) against a shadow model of live
requests and checks block-pool accounting invariants after every rule.

Every rule replays the same operation on two managers built from the same
config, one with prefix caching on and one off (same-trace differential).
The cache-off side must never report a hit, and token progress must match
on both sides while a request is live on both and never decoded
asymmetrically (once a request decodes while live on one side only, its
two copies' token streams differ by construction). (Allocation outcomes may
diverge: shared prefix blocks stay pinned by other live requests on the
cache-on side, so neither side's success implies the other's.) On the
cache-on side a reference model tracks which block hashes could ever have
been cached and which are definitely still cached (held by live requests):
the reported hit must stay within those bounds. For the hybrid (sliding
window + full attention) layout the lower bound is checked on the
full-attention group only, because the reported aggregate hit is the min
across groups and the sliding-window group legitimately caches fewer
blocks than the reference tracks.
"""

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
    run_state_machine_as_test,
)

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_config_hybrid_model,
    make_kv_cache_manager,
    make_request,
)
from tests.v1.fuzz.fuzz_utils import VOCAB_SIZE, assert_block_pool_consistent
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

NUM_BLOCKS = 64
MAX_TRACKED_REQUESTS = 16


class _Side:
    """One manager of the differential pair plus its shadow request state."""

    def __init__(self, manager: KVCacheManager):
        self.manager = manager
        # Requests with allocated blocks.
        self.live: dict[str, Request] = {}
        # Requests that failed allocation and hold no blocks.
        self.waiting: dict[str, Request] = {}


class KVCacheManagerFuzzStateMachine(RuleBasedStateMachine):
    """Same-trace allocate/decode/free traffic against two KVCacheManagers."""

    hybrid: bool = False
    block_size: int = 16

    def __init__(self):
        super().__init__()
        init_none_hash(sha256)

        def make_manager(enable_caching: bool) -> KVCacheManager:
            if self.hybrid:
                kv_cache_config = make_kv_cache_config_hybrid_model(
                    self.block_size, NUM_BLOCKS, sliding_window_blocks=4
                )
            else:
                kv_cache_config = make_kv_cache_config(self.block_size, NUM_BLOCKS)
            return make_kv_cache_manager(
                kv_cache_config,
                max_model_len=8192,
                enable_caching=enable_caching,
                hash_block_size=self.block_size,
            )

        self.on = _Side(make_manager(enable_caching=True))
        self.off = _Side(make_manager(enable_caching=False))
        # Drawn lazily on first use (no draw source in __init__).
        self.shared_prefix: list[int] | None = None
        self.max_output: dict[str, int] = {}
        self.next_id = 0
        # Requests that ever decoded while live on only one side; their two
        # copies' token streams differ by construction afterwards.
        self.diverged: set[str] = set()
        # Reference model for cache-on hit bounds.
        self.ref_ever_cached: set = set()
        self.ref_live_cached: dict[str, set] = {}

    def _ref_mark_cached(self, request: Request, cached_tokens: int):
        """Record the block hashes the manager just had the chance to cache."""
        num_blocks = min(cached_tokens, request.num_tokens) // self.block_size
        hashes = set(request.block_hashes[:num_blocks])
        self.ref_ever_cached |= hashes
        self.ref_live_cached.setdefault(request.request_id, set()).update(hashes)

    def _check_hit_bounds(self, request: Request, num_computed_tokens: int):
        """Check a cache-on get_computed_blocks hit against the reference."""
        block_size = self.block_size
        assert num_computed_tokens % block_size == 0
        # The manager caps hits at num_tokens - 1 (last token is recomputed).
        cap = (request.num_tokens - 1) // block_size * block_size
        upper_run = 0
        for block_hash in request.block_hashes:
            if block_hash not in self.ref_ever_cached:
                break
            upper_run += 1
        upper = min(upper_run * block_size, cap)
        assert num_computed_tokens <= upper, (
            f"hit {num_computed_tokens} exceeds ever-cached bound {upper}"
        )
        alive: set = set()
        for hashes in self.ref_live_cached.values():
            alive |= hashes
        lower_run = 0
        for block_hash in request.block_hashes:
            if block_hash not in alive:
                break
            lower_run += 1
        lower = min(lower_run * block_size, cap)
        if self.hybrid:
            # The aggregate hit is the min across groups; the sliding-window
            # group legitimately caches fewer blocks than the reference
            # tracks, so the lower bound only applies to the full-attention
            # group's own hit.
            coordinator = self.on.manager.coordinator
            assert coordinator.full_attention_group_id is not None
            _, per_group_hits = coordinator.find_longest_cache_hit_per_group(
                request.block_hashes, request.num_tokens - 1
            )
            fa_hit = per_group_hits[coordinator.full_attention_group_id]
            assert fa_hit >= lower, (
                f"full-attention hit {fa_hit} below definitely-cached bound {lower}"
            )
        else:
            assert num_computed_tokens >= lower, (
                f"hit {num_computed_tokens} below definitely-cached bound {lower}"
            )

    def _try_allocate(self, side: _Side, request: Request) -> bool:
        computed_blocks, num_computed_tokens, _ = side.manager.get_computed_blocks(
            request
        )
        if side is self.on:
            self._check_hit_bounds(request, num_computed_tokens)
        else:
            # Cache-off differential: a hit here would mean cache state
            # leaked into a caching-disabled manager.
            assert num_computed_tokens == 0
        num_new_tokens = request.num_tokens - num_computed_tokens
        new_blocks = side.manager.allocate_slots(
            request, num_new_tokens, num_computed_tokens, computed_blocks
        )
        if new_blocks is None:
            side.waiting[request.request_id] = request
            return False
        request.num_computed_tokens = request.num_tokens
        if side is self.on:
            self._ref_mark_cached(request, request.num_tokens)
        side.live[request.request_id] = request
        side.waiting.pop(request.request_id, None)
        return True

    def _drop_live(self, side: _Side, request_id: str):
        side.live.pop(request_id, None)
        if side is self.on:
            self.ref_live_cached.pop(request_id, None)

    def _live_ids(self) -> list[str]:
        return sorted(set(self.on.live) | set(self.off.live))

    def _waiting_ids(self) -> list[str]:
        return sorted(set(self.on.waiting) | set(self.off.waiting))

    @rule(data=st.data())
    def act(self, data: st.DataObject):
        # One rule with the action as an explicit draw. Under a derandomized
        # stream, hypothesis' per-rule swarm flags phase-lock on identical
        # steps and can disable a rule for the whole run (observed on the
        # scheduler fuzz machine: a rule disabled 47/47 evaluations). A drawn
        # action keeps every action reachable on every stream and stays
        # shrinkable.
        action = data.draw(st.integers(1, 10))
        can_submit = len(self.on.live) + len(self.on.waiting) < MAX_TRACKED_REQUESTS
        if action <= 3 and can_submit:
            self._submit_request(data)
        elif action <= 5 and self._waiting_ids():
            self._retry_waiting_request(data)
        elif action <= 9 and self._live_ids():
            request_id = data.draw(st.sampled_from(self._live_ids()))
            if action <= 7:
                self._decode_token(data, request_id)
            elif action == 8:
                self._free_request(request_id)
            else:
                self._recache_request(request_id)
        else:
            self._reset_prefix_cache()

    def _submit_request(self, data: st.DataObject):
        if self.shared_prefix is None:
            blob = data.draw(
                st.binary(min_size=2 * self.block_size, max_size=2 * self.block_size)
            )
            self.shared_prefix = [b % VOCAB_SIZE for b in blob]
        num_tail_tokens = data.draw(st.integers(1, 3 * self.block_size))
        blob = data.draw(st.binary(min_size=num_tail_tokens, max_size=num_tail_tokens))
        tail = [b % VOCAB_SIZE for b in blob]
        use_shared = data.draw(st.booleans())
        prompt = (self.shared_prefix + tail) if use_shared else tail
        request_id = str(self.next_id)
        self.next_id += 1
        self.max_output[request_id] = data.draw(st.integers(1, 32))
        self._try_allocate(
            self.on, make_request(request_id, prompt, self.block_size, sha256)
        )
        self._try_allocate(
            self.off, make_request(request_id, prompt, self.block_size, sha256)
        )

    def _retry_waiting_request(self, data: st.DataObject):
        request_id = data.draw(st.sampled_from(self._waiting_ids()))
        if (on_pending := self.on.waiting.get(request_id)) is not None:
            self._try_allocate(self.on, on_pending)
        if (off_pending := self.off.waiting.get(request_id)) is not None:
            self._try_allocate(self.off, off_pending)

    def _decode_token(self, data: st.DataObject, request_id: str):
        token = data.draw(st.integers(0, VOCAB_SIZE - 1))
        ref_request = self.on.live.get(request_id) or self.off.live[request_id]
        complete = (
            len(ref_request.output_token_ids) >= self.max_output[request_id]
            or data.draw(st.integers(1, 20)) == 1
        )
        if (request_id in self.on.live) != (request_id in self.off.live):
            self.diverged.add(request_id)
        for side in (self.on, self.off):
            request = side.live.get(request_id)
            if request is None:
                continue
            if complete:
                side.manager.free(request)
                self._drop_live(side, request_id)
                continue
            request.append_output_token_ids(token)
            request.num_computed_tokens += 1
            if side.manager.allocate_slots(request, 1) is None:
                # Out of blocks mid-decode: drop like a preemption.
                side.manager.free(request)
                self._drop_live(side, request_id)
            elif side is self.on:
                self._ref_mark_cached(request, request.num_tokens)
        if (request_id in self.on.live) != (request_id in self.off.live):
            self.diverged.add(request_id)
        # Same trace on both sides: while a request is live on both and
        # never decoded asymmetrically, its token progress is identical.
        req_on = self.on.live.get(request_id)
        req_off = self.off.live.get(request_id)
        if (
            req_on is not None
            and req_off is not None
            and request_id not in self.diverged
        ):
            assert req_on.num_computed_tokens == req_off.num_computed_tokens

    def _free_request(self, request_id: str):
        for side in (self.on, self.off):
            request = side.live.get(request_id)
            if request is not None:
                side.manager.free(request)
                self._drop_live(side, request_id)

    def _recache_request(self, request_id: str):
        for side in (self.on, self.off):
            request = side.live.get(request_id)
            if request is None:
                continue
            side.manager.cache_blocks(request, request.num_computed_tokens)
            if side is self.on:
                self._ref_mark_cached(request, request.num_computed_tokens)

    def _reset_prefix_cache(self):
        # Reset only succeeds when no request holds blocks.
        for side in (self.on, self.off):
            if side.manager.reset_prefix_cache():
                assert not side.live
                if side is self.on:
                    self.ref_ever_cached.clear()
            else:
                assert side.live

    @invariant()
    def check_block_pool_consistency(self):
        for side in (self.on, self.off):
            block_pool = side.manager.block_pool
            assert_block_pool_consistent(block_pool)
            holder_count: dict[int, int] = {}
            for request_id in side.live:
                for group_block_ids in side.manager.get_block_ids(request_id):
                    for block_id in group_block_ids:
                        block = block_pool.blocks[block_id]
                        if block.is_null:
                            continue
                        holder_count[block_id] = holder_count.get(block_id, 0) + 1
            for block_id, count in holder_count.items():
                block = block_pool.blocks[block_id]
                assert block.ref_cnt == count, (
                    f"block {block_id}: ref_cnt {block.ref_cnt} != holders {count}"
                )

    def teardown(self):
        for side in (self.on, self.off):
            for request in side.live.values():
                side.manager.free(request)
            side.live.clear()
            block_pool = side.manager.block_pool
            assert block_pool.get_num_free_blocks() == block_pool.num_gpu_blocks - 1
            assert_block_pool_consistent(block_pool)
            assert side.manager.reset_prefix_cache()


@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("hybrid", [False, True], ids=["full", "hybrid-swa"])
def test_kv_cache_manager_fuzz(hybrid: bool, block_size: int):
    machine_cls = type(
        "BoundKVCacheManagerFuzzStateMachine",
        (KVCacheManagerFuzzStateMachine,),
        {
            "hybrid": hybrid,
            "block_size": block_size,
        },
    )
    run_state_machine_as_test(machine_cls)
