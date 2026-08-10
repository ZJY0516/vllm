# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared harness for hypothesis-based fuzz tests of the v1 core."""

import os
import uuid
from collections.abc import Mapping

from hypothesis import settings
from hypothesis import strategies as st

from tests.v1.core.utils import EOS_TOKEN_ID
from vllm.config import VllmConfig
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request

VOCAB_SIZE = 100
# Outside the regular sample range so it only fires as an explicit stop token.
STOP_TOKEN_ID = VOCAB_SIZE

# Fast deterministic profile for CI smoke; "long" is activated via
# HYPOTHESIS_PROFILE=long for deeper local runs.
settings.register_profile(
    "ci",
    max_examples=3,
    stateful_step_count=40,
    deadline=None,
    derandomize=True,
)
settings.register_profile(
    "long",
    max_examples=30,
    stateful_step_count=400,
    deadline=None,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


def draw_token_ids(data: st.DataObject, num_tokens: int) -> list[int]:
    """Draw token ids as a single blob so traces shrink to small byte strings."""
    blob = data.draw(st.binary(min_size=num_tokens, max_size=num_tokens))
    return [b % VOCAB_SIZE for b in blob]


def random_request(
    data: st.DataObject,
    vllm_config: VllmConfig,
    mm_pool: list[tuple[str, int]],
    max_tokens_range: tuple[int, int] = (1, 500),
    num_tokens_range: tuple[int, int] = (1, 500),
    arrival_time_range: tuple[float, float] = (0.0, 1.0),
    priority_range: tuple[int, int] = (-3, 3),
    num_mm_item_range: tuple[int, int] = (0, 2),
    shared_prefix: list[int] | None = None,
    shared_prefix_prob: float = 0.0,
    stop_token_ids: list[int] | None = None,
    request_id: str | None = None,
) -> Request:
    """Draw a random Request, optionally sharing a token prefix with others.

    All randomness is drawn from ``data`` so hypothesis controls (and can
    shrink) the full trace.

    Args:
        data: Hypothesis draw source owned by the calling rule.
        vllm_config: Config of the scheduler under test (for the block hasher).
        mm_pool: (identifier, token length) pairs to sample mm items from.
            Identifiers carry no request id, so encoder cache hits across
            requests actually happen. The token length is fixed per
            identifier, mirroring content-addressed real inputs. Items that
            overlap or exceed the drawn prompt are skipped.
        max_tokens_range: Range for sampling_params.max_tokens.
        num_tokens_range: Range for the (non-shared) prompt length.
        arrival_time_range: Range for the arrival time.
        priority_range: Range for the scheduling priority.
        num_mm_item_range: Range for the number of dummy mm items.
        shared_prefix: Optional block-aligned token ids reused across requests
            to trigger prefix cache hits.
        shared_prefix_prob: Probability of prepending ``shared_prefix``.
        stop_token_ids: Optional stop token ids for the sampling params.
        request_id: Optional explicit request id. State-machine tests should
            pass deterministic ids: ``Request.__lt__`` breaks priority ties by
            request id, so random uuids make replays diverge.

    Returns:
        A new Request with a unique request id.
    """
    max_tokens = data.draw(st.integers(*max_tokens_range))
    num_tokens = data.draw(st.integers(*num_tokens_range))
    priority = data.draw(st.integers(*priority_range))
    arrival_time = data.draw(st.floats(*arrival_time_range))
    num_mm_item = data.draw(st.integers(*num_mm_item_range))

    prompt_token_ids = draw_token_ids(data, num_tokens)
    if shared_prefix and data.draw(st.floats(0.0, 1.0)) < shared_prefix_prob:
        prompt_token_ids = list(shared_prefix) + prompt_token_ids

    request_id = request_id if request_id is not None else uuid.uuid4().hex

    mm_features = []
    prev_mm_end = 0
    num_items = min(num_mm_item, len(prompt_token_ids))
    if num_items:
        mm_starts = data.draw(
            st.lists(
                st.integers(0, len(prompt_token_ids) - 1),
                min_size=num_items,
                max_size=num_items,
            )
        )
        for mm_start in sorted(mm_starts):
            identifier, mm_length = mm_pool[data.draw(st.integers(0, len(mm_pool) - 1))]
            if mm_start < prev_mm_end or mm_start + mm_length > len(prompt_token_ids):
                continue
            prev_mm_end = mm_start + mm_length
            mm_features.append(
                MultiModalFeatureSpec(
                    data=MultiModalKwargsItem.dummy(),
                    mm_position=PlaceholderRange(offset=mm_start, length=mm_length),
                    identifier=identifier,
                    modality="image",
                )
            )

    sampling_params = SamplingParams(
        ignore_eos=False, max_tokens=max_tokens, stop_token_ids=stop_token_ids
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    caching_hash_fn = get_hash_fn_by_name(
        vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)
    block_hasher = get_request_block_hasher(
        vllm_config.cache_config.block_size, caching_hash_fn
    )

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=mm_features if mm_features else None,
        arrival_time=arrival_time,
        priority=priority,
        block_hasher=block_hasher,
    )


def mock_execute_model(
    scheduler_output: SchedulerOutput,
    requests: Mapping[str, Request],
    data: st.DataObject,
) -> ModelRunnerOutput:
    """Mock model execution following the real runner contract.

    A request only samples when this step computed its last prompt token:
    requests mid chunked prefill get an empty ``sampled_token_ids`` entry
    (see ``Scheduler._update_request_with_output``). Every other entry holds
    ``1 + num_accepted`` tokens, where ``num_accepted`` never exceeds the
    drafts actually scheduled for that request in
    ``scheduler_output.scheduled_spec_decode_tokens`` -- the scheduler
    derives the rejection count from the entry length, so over-counting
    corrupts ``num_computed_tokens`` and under-counting never exercises
    rejections. EOS / stop tokens only appear in the final position, and a
    stop token only when the request configured it.
    """
    request_ids: list[str] = [req.req_id for req in scheduler_output.scheduled_new_reqs]
    request_ids.extend(scheduler_output.scheduled_cached_reqs.req_ids)

    sampled_token_ids: list[list[int]] = []
    for req_id in request_ids:
        request = requests[req_id]
        if request.is_prefill_chunk:
            sampled_token_ids.append([])
            continue
        num_drafts = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
        num_accepted = data.draw(st.integers(0, num_drafts)) if num_drafts else 0
        token_ids = draw_token_ids(data, 1 + num_accepted)
        sampling_params = request.sampling_params
        stop_token_ids = sampling_params.stop_token_ids if sampling_params else None
        stop_roll = data.draw(st.integers(1, 20))
        if stop_roll == 1:
            token_ids[-1] = EOS_TOKEN_ID
        elif stop_roll == 2 and stop_token_ids and STOP_TOKEN_ID in stop_token_ids:
            token_ids[-1] = STOP_TOKEN_ID
        sampled_token_ids.append(token_ids)

    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(request_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def mock_draft_token_ids(
    scheduler_output: SchedulerOutput,
    data: st.DataObject,
    num_speculative_tokens: int,
    seen_request_prompt_length: dict[str, int],
) -> DraftTokenIds:
    """Mock draft tokens for requests past prefill.

    Mirrors ``_mock_draft_token_ids`` from test_priority_scheduler_random.py.
    ``seen_request_prompt_length`` must be shared across all calls for a
    given scheduler run.
    """
    request_ids: list[str] = []
    sampled_token_ids: list[list[int]] = []
    for request in scheduler_output.scheduled_new_reqs:
        assert request.req_id not in seen_request_prompt_length
        seen_request_prompt_length[request.req_id] = len(request.prompt_token_ids or [])
        if request.num_computed_tokens >= seen_request_prompt_length[request.req_id]:
            request_ids.append(request.req_id)
            num_tokens = data.draw(st.integers(0, num_speculative_tokens))
            sampled_token_ids.append(draw_token_ids(data, num_tokens))
    for req_id, num_computed_tokens in zip(
        scheduler_output.scheduled_cached_reqs.req_ids,
        scheduler_output.scheduled_cached_reqs.num_computed_tokens,
    ):
        if num_computed_tokens >= seen_request_prompt_length[req_id]:
            request_ids.append(req_id)
            num_tokens = data.draw(st.integers(0, num_speculative_tokens))
            sampled_token_ids.append(draw_token_ids(data, num_tokens))
    return DraftTokenIds(req_ids=request_ids, draft_token_ids=sampled_token_ids)


def assert_scheduler_output_valid(
    scheduler_output: SchedulerOutput,
    seen_request_ids: set[str],
    seen_mm_hashes: set[str],
) -> None:
    """Check SchedulerOutput internal consistency.

    Same logic as ``_check_valid_scheduler_output`` from
    test_priority_scheduler_random.py. ``seen_request_ids`` and
    ``seen_mm_hashes`` must be shared across all calls for a given run.
    """
    for req in scheduler_output.scheduled_new_reqs:
        assert req.req_id not in seen_request_ids
        seen_request_ids.add(req.req_id)
    for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
        assert req_id in seen_request_ids

    req_ids = set[str]()
    req_ids.update(req.req_id for req in scheduler_output.scheduled_new_reqs)
    req_ids.update(scheduler_output.scheduled_cached_reqs.req_ids)

    assert set(scheduler_output.num_scheduled_tokens.keys()) == req_ids
    assert (
        sum(scheduler_output.num_scheduled_tokens.values())
        == scheduler_output.total_num_scheduled_tokens
    )

    assert set(scheduler_output.scheduled_spec_decode_tokens.keys()) <= req_ids
    assert set(scheduler_output.scheduled_encoder_inputs.keys()) <= req_ids

    for req in scheduler_output.scheduled_new_reqs:
        for mm_feature in req.mm_features:
            seen_mm_hashes.add(mm_feature.identifier)
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        assert mm_hash in seen_mm_hashes

    assert scheduler_output.finished_req_ids <= seen_request_ids


def assert_block_pool_consistent(block_pool: BlockPool) -> None:
    """Check BlockPool accounting invariants.

    Every non-null block is either in the free queue (ref_cnt == 0) or held
    (ref_cnt > 0), never both or neither, and the free queue has no
    duplicates. Every cached hash maps to a valid block whose own hash
    metadata points back at the hash.
    """
    free_blocks = block_pool.free_block_queue.get_all_free_blocks()
    free_ids = [block.block_id for block in free_blocks]
    assert len(free_ids) == len(set(free_ids)), "duplicate block in free queue"
    assert len(free_ids) == block_pool.get_num_free_blocks()
    free_id_set = set(free_ids)
    for block in block_pool.blocks:
        if block.is_null or block.ref_cnt > 0:
            assert block.block_id not in free_id_set
        else:
            assert block.block_id in free_id_set

    cached = block_pool.cached_block_hash_to_block
    for key, blocks in list(cached._cache.items()):
        block_list = [blocks] if isinstance(blocks, KVCacheBlock) else blocks.values()
        for block in block_list:
            assert 0 <= block.block_id < block_pool.num_gpu_blocks
            assert not block.is_null
            extra_hashes = block_pool.cached_block_hashes_by_block.get(
                block.block_id, set()
            )
            assert block.block_hash == key or key in extra_hashes


def assert_encoder_cache_consistent(manager: EncoderCacheManager) -> None:
    """Check EncoderCacheManager accounting invariants.

    Freeable capacity is exactly the free slots plus the evictable
    (unreferenced) entries; the freeable map is precisely the set of cached
    entries with an empty reference set; every cache-side reference is
    backed by a request-side reference list.
    """
    assert 0 <= manager.num_free_slots <= manager.cache_size
    assert manager.num_freeable_slots == manager.num_free_slots + sum(
        manager.freeable.values()
    )
    assert manager.num_freeable_slots <= manager.cache_size
    assert set(manager.freeable) == {
        mm_hash for mm_hash, refs in manager.cached.items() if not refs
    }
    ref_count: dict[str, int] = {}
    for refs in manager.cached.values():
        for req_id in refs:
            ref_count[req_id] = ref_count.get(req_id, 0) + 1
    for input_ids in manager.request_cached_ids.values():
        assert input_ids
    # Only one direction holds: a request with two inputs sharing one
    # identifier keeps a request-side record for the later input after
    # freeing the earlier one already dropped the entry's only cache-side
    # reference (the reference set is per request, not per input), so the
    # request side is a superset of the cache side.
    assert set(ref_count) <= set(manager.request_cached_ids)


def assert_scheduler_invariants(
    scheduler: Scheduler, allow_in_flight: bool = False
) -> None:
    """Check scheduler / block pool invariants after a completed step.

    Running requests hold blocks with ref_cnt exactly equal to the number of
    running holders (prefix caching makes blocks shareable), and each
    request's computed token count stays within its token list.

    Args:
        scheduler: The scheduler under test.
        allow_in_flight: Allow num_computed_tokens to run ahead of the token
            list by num_in_flight_tokens (async scheduling counts in-flight
            tokens optimistically).
    """
    block_pool = scheduler.kv_cache_manager.block_pool
    assert_block_pool_consistent(block_pool)

    holder_count: dict[int, int] = {}
    for request in scheduler.running:
        num_tokens_bound = len(request.all_token_ids)
        if allow_in_flight:
            num_tokens_bound += request.num_in_flight_tokens
        assert 0 <= request.num_computed_tokens <= num_tokens_bound
        block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
        for group_block_ids in block_ids:
            for block_id in group_block_ids:
                block = block_pool.blocks[block_id]
                if block.is_null:
                    continue
                assert block.ref_cnt >= 1
                holder_count[block_id] = holder_count.get(block_id, 0) + 1
    for block_id, count in holder_count.items():
        assert block_pool.blocks[block_id].ref_cnt == count


def assert_pool_drained(scheduler: Scheduler) -> None:
    """Check that no request state or blocks leak after draining.

    All requests must be finished and every block except the null block must
    be back in the free queue (cached blocks stay there as eviction
    candidates).
    """
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    block_pool = scheduler.kv_cache_manager.block_pool
    assert block_pool.get_num_free_blocks() == block_pool.num_gpu_blocks - 1
    assert_block_pool_consistent(block_pool)
