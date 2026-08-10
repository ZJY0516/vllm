# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hypothesis state-machine fuzz for the v1 priority scheduler.

Drives random interleavings of add / schedule+execute / abort / cache-reset
against ``Scheduler`` and checks safety invariants (block-pool and
encoder-cache accounting) plus a liveness invariant (committed progress
within a bounded window) after every step. All inputs -- request shapes,
token contents, mm layout, mock model outputs -- are hypothesis draws, so
failures shrink to minimal traces.
"""

from collections import deque
from typing import Any

import pytest
import torch
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    rule,
    run_state_machine_as_test,
)

import vllm.envs as envs
from tests.v1.core.test_scheduler import create_scheduler_with_priority
from tests.v1.fuzz.fuzz_utils import (
    STOP_TOKEN_ID,
    assert_encoder_cache_consistent,
    assert_pool_drained,
    assert_scheduler_invariants,
    assert_scheduler_output_valid,
    draw_token_ids,
    mock_draft_token_ids,
    mock_execute_model,
    random_request,
)
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

# (max_input_tokens, max_output_tokens, max_num_seqs, num_blocks)
PROFILES = {
    "standard": (5000, 500, 256, 10000),
    "preemption": (500, 5000, 1024, 1000),
}

MAX_ACTIVE_REQUESTS = 64
# Consecutive steps without committed progress (finished request, delivered
# output token, first completed prefill) tolerated while work is pending
# before failing. Async scheduling adds one step of in-flight latency; the
# window stays far below any livelock.
MAX_ZERO_PROGRESS_STREAK = 10
# Upper bound for the drawn encoder cache size in stress configs; the lower
# bound of 1 makes single-embed items contend for every slot.
MAX_ENCODER_CACHE_SIZE = 64

# Qwen2.5-VL-style image placeholder length (3 blocks) for the hybrid layout.
HYBRID_MM_ITEM_LENGTH = 48


def create_hybrid_priority_scheduler(
    enable_prefix_caching: bool,
    num_speculative_tokens: int | None,
    max_num_seqs: int,
    num_blocks: int,
    block_size: int = 16,
    async_scheduling: bool = False,
) -> Scheduler:
    """Create a priority scheduler with Qwen3.5-style hybrid KV cache groups.

    Two groups: full attention + GDN (MambaSpec). Mirrors MambaModelConfig:
    the mamba group uses cache mode "align" when prefix caching is enabled
    and "none" otherwise, with the mamba block size aligned to block_size.
    """
    model_config = ModelConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy="priority",
        async_scheduling=async_scheduling,
        # Ensure admission/preemption mechanics are deterministic
        watermark=0.0,
    )
    mamba_cache_mode = "align" if enable_prefix_caching else "none"
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode=mamba_cache_mode,
    )
    speculative_config: SpeculativeConfig | None = None
    if num_speculative_tokens is not None:
        speculative_config = SpeculativeConfig(
            model="ngram", num_speculative_tokens=num_speculative_tokens
        )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        speculative_config=speculative_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["gdn"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((16, 64),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode=mamba_cache_mode,
                    num_speculative_blocks=num_speculative_tokens or 0,
                ),
            ),
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
        log_stats=True,
    )
    scheduler.use_v2_model_runner = bool(envs.VLLM_USE_V2_MODEL_RUNNER)
    return scheduler


class SchedulerFuzzStateMachine(RuleBasedStateMachine):
    """Random add/step/abort/reset traffic against a priority scheduler."""

    enable_prefix_caching: bool = True
    num_speculative_tokens: int | None = None
    profile: tuple[int, int, int, int] = PROFILES["standard"]
    # "full": single FullAttentionSpec group; "hybrid": Qwen3.5-style
    # full attention + GDN (MambaSpec) groups with heavier mm pressure.
    kv_layout: str = "full"
    block_size: int = 16
    # When set, draw a small encoder cache size to stress encoder scheduling.
    stress_encoder_cache: bool = False
    async_scheduling: bool = False

    def __init__(self):
        super().__init__()
        max_input_tokens, max_output_tokens, max_num_seqs, num_blocks = self.profile
        self.max_tokens_range = (1, max_output_tokens)
        self.num_tokens_range = (1, max_input_tokens)
        self.num_mm_item_range = (
            (0, 4)
            if self.stress_encoder_cache or self.kv_layout == "hybrid"
            else (0, 2)
        )
        if self.kv_layout == "hybrid":
            self.scheduler = create_hybrid_priority_scheduler(
                enable_prefix_caching=self.enable_prefix_caching,
                num_speculative_tokens=self.num_speculative_tokens,
                max_num_seqs=max_num_seqs,
                num_blocks=num_blocks,
                block_size=self.block_size,
                async_scheduling=self.async_scheduling,
            )
        else:
            self.scheduler = create_scheduler_with_priority(
                model="Qwen/Qwen2.5-VL-3B-Instruct",
                max_num_seqs=max_num_seqs,
                enable_prefix_caching=self.enable_prefix_caching,
                num_blocks=num_blocks,
                num_speculative_tokens=self.num_speculative_tokens,
                block_size=self.block_size,
                async_scheduling=self.async_scheduling,
            )
        self.active_request_ids: set[str] = set()
        self.seen_request_ids: set[str] = set()
        self.seen_mm_hashes: set[str] = set()
        self.seen_request_prompt_length: dict[str, int] = {}
        self.next_request_id = 0
        # Outputs scheduled but not yet processed (async scheduling only).
        self.pending_outputs: deque = deque()
        # Committed-progress liveness bookkeeping.
        self.committed_progress = 0
        self.progress_mark = 0
        self.zero_progress_streak = 0
        self.known_output_lengths: dict[str, int] = {}
        self.prefill_completed: set[str] = set()

    @initialize(data=st.data())
    def draw_shared_layout(self, data: st.DataObject):
        """Draw per-run shared inputs: mm item pool and shared token prefix."""
        if self.stress_encoder_cache:
            encoder_cache_size = data.draw(st.integers(1, MAX_ENCODER_CACHE_SIZE))
            self.scheduler.encoder_cache_manager = EncoderCacheManager(
                cache_size=encoder_cache_size
            )
            self.scheduler.max_num_encoder_input_tokens = encoder_cache_size
            mm_item_length_range = (1, encoder_cache_size)
        elif self.kv_layout == "hybrid":
            mm_item_length_range = (HYBRID_MM_ITEM_LENGTH,) * 2
        else:
            mm_item_length_range = (10, 10)
        pool_size = data.draw(st.integers(2, 8))
        self.mm_pool = [
            (f"mmhash-{i}", data.draw(st.integers(*mm_item_length_range)))
            for i in range(pool_size)
        ]
        block_size = self.scheduler.cache_config.block_size
        prefix_len = data.draw(st.integers(1, 4)) * block_size
        self.shared_prefix = draw_token_ids(data, prefix_len)

    @rule(data=st.data())
    def step(self, data: st.DataObject):
        # One rule with the action as an explicit draw. Under a derandomized
        # stream, hypothesis' per-rule swarm flags phase-lock on identical
        # empty steps and can disable a rule for the whole run (observed:
        # add_requests disabled 47/47 evaluations, so CI never added a single
        # request). A drawn action keeps every action reachable on every
        # stream and stays shrinkable.
        action = data.draw(st.integers(1, 10))
        if action <= 2 and len(self.active_request_ids) < MAX_ACTIVE_REQUESTS:
            self._add_requests(data)
        elif action == 3 and self.active_request_ids:
            self._abort_random_request(data)
        elif action == 4:
            # Contract: reset only succeeds when no running request holds
            # blocks.
            assert self.scheduler.reset_prefix_cache() == (not self.scheduler.running)
        else:
            self._schedule_step(data)

    def _add_requests(self, data: st.DataObject):
        for _ in range(data.draw(st.integers(1, 2))):
            stop_token_ids = [STOP_TOKEN_ID] if data.draw(st.booleans()) else None
            request = random_request(
                data,
                self.scheduler.vllm_config,
                mm_pool=self.mm_pool,
                max_tokens_range=self.max_tokens_range,
                num_tokens_range=self.num_tokens_range,
                num_mm_item_range=self.num_mm_item_range,
                shared_prefix=self.shared_prefix,
                shared_prefix_prob=0.6,
                stop_token_ids=stop_token_ids,
                request_id=f"req-{self.next_request_id}",
            )
            self.next_request_id += 1
            self.scheduler.add_request(request)
            self.active_request_ids.add(request.request_id)

    def _schedule_step(self, data: st.DataObject):
        active_before = set(self.active_request_ids)
        scheduler_output = self.scheduler.schedule()
        assert_scheduler_output_valid(
            scheduler_output, self.seen_request_ids, self.seen_mm_hashes
        )
        model_output = mock_execute_model(
            scheduler_output, self.scheduler.requests, data
        )
        if self.async_scheduling:
            # Real in-flight overlap: the output scheduled one step ago is
            # processed only now, so it was still in flight during the
            # schedule() above (stale outputs and abort/preemption of
            # in-flight requests become reachable).
            if self.pending_outputs:
                old_output, old_model_output = self.pending_outputs.popleft()
                self.scheduler.update_from_output(old_output, old_model_output)
            if scheduler_output.total_num_scheduled_tokens > 0:
                self.pending_outputs.append((scheduler_output, model_output))
        else:
            self.scheduler.update_from_output(scheduler_output, model_output)
            if self.num_speculative_tokens is not None:
                self.scheduler.update_draft_token_ids(
                    mock_draft_token_ids(
                        scheduler_output,
                        data,
                        self.num_speculative_tokens,
                        self.seen_request_prompt_length,
                    )
                )
        self._refresh_active()
        num_finished = len(active_before - self.active_request_ids)
        self._update_committed_progress(num_finished)
        self._check_liveness()
        assert_scheduler_invariants(
            self.scheduler, allow_in_flight=self.async_scheduling
        )
        assert_encoder_cache_consistent(self.scheduler.encoder_cache_manager)

    def _abort_random_request(self, data: st.DataObject):
        request_id = data.draw(st.sampled_from(sorted(self.active_request_ids)))
        self.scheduler.finish_requests([request_id], RequestStatus.FINISHED_ABORTED)
        self.active_request_ids.discard(request_id)
        # Aborted requests may be reported as finished without ever having
        # been scheduled; the harness knows them, so register them as seen.
        self.seen_request_ids.add(request_id)
        assert_scheduler_invariants(
            self.scheduler, allow_in_flight=self.async_scheduling
        )
        assert_encoder_cache_consistent(self.scheduler.encoder_cache_manager)

    def _update_committed_progress(self, num_finished: int):
        """Accumulate progress that cannot be rolled back."""
        self.committed_progress += num_finished
        for req_id, request in self.scheduler.requests.items():
            delivered = len(request.output_token_ids)
            previous = self.known_output_lengths.get(req_id, 0)
            if delivered > previous:
                self.committed_progress += delivered - previous
                self.known_output_lengths[req_id] = delivered
            # Preemption resets num_computed_tokens, but a first completed
            # prefill still proves the request can get through; in-flight
            # tokens (async scheduling) are not committed yet.
            committed = request.num_computed_tokens - request.num_in_flight_tokens
            if req_id not in self.prefill_completed and committed >= request.num_tokens:
                self.prefill_completed.add(req_id)
                self.committed_progress += 1

    def _check_liveness(self):
        pending_work = bool(self.scheduler.waiting) or bool(self.scheduler.running)
        if pending_work and self.committed_progress == self.progress_mark:
            self.zero_progress_streak += 1
        else:
            self.zero_progress_streak = 0
        self.progress_mark = self.committed_progress
        assert self.zero_progress_streak <= MAX_ZERO_PROGRESS_STREAK, (
            f"no committed progress for {self.zero_progress_streak} "
            f"consecutive steps with pending work; {self._work_summary()}"
        )

    def _work_summary(self) -> str:
        def fmt(request) -> str:
            return (
                f"{request.request_id}(status={request.status.name}, "
                f"computed={request.num_computed_tokens}/{request.num_tokens}, "
                f"mm={len(request.mm_features)})"
            )

        running = ", ".join(fmt(r) for r in list(self.scheduler.running)[:5])
        waiting = ", ".join(fmt(r) for r in list(self.scheduler.waiting)[:5])
        return f"running: [{running}] waiting: [{waiting}]"

    def _refresh_active(self):
        for request_id in list(self.active_request_ids):
            request = self.scheduler.requests.get(request_id)
            if request is None or request.is_finished():
                self.active_request_ids.discard(request_id)

    def teardown(self):
        while self.pending_outputs:
            scheduler_output, model_output = self.pending_outputs.popleft()
            self.scheduler.update_from_output(scheduler_output, model_output)
        self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
        assert_pool_drained(self.scheduler)
        encoder_manager = self.scheduler.encoder_cache_manager
        assert_encoder_cache_consistent(encoder_manager)
        # No references remain, so every cached embed is immediately
        # evictable; anything less means a reference leaked.
        assert encoder_manager.num_freeable_slots == encoder_manager.cache_size


def _sched_configs() -> list[dict[str, Any]]:
    configs = []
    for layout in ("full", "hybrid"):
        for pc in (True, False):
            for spec in (None, 1, 5):
                for profile_name, profile in PROFILES.items():
                    configs.append(
                        dict(
                            kv_layout=layout,
                            enable_prefix_caching=pc,
                            num_speculative_tokens=spec,
                            profile=profile,
                            profile_name=profile_name,
                        )
                    )
    # Drawn-tight encoder cache: one item fits, several often do not.
    for layout in ("full", "hybrid"):
        for pc in (True, False):
            for spec in (None, 1, 5):
                configs.append(
                    dict(
                        kv_layout=layout,
                        enable_prefix_caching=pc,
                        num_speculative_tokens=spec,
                        profile=PROFILES["standard"],
                        profile_name="standard",
                        stress_encoder_cache=True,
                    )
                )
    # Block size variants.
    for layout in ("full", "hybrid"):
        for block_size in (8, 32):
            configs.append(
                dict(
                    kv_layout=layout,
                    enable_prefix_caching=True,
                    num_speculative_tokens=None,
                    profile=PROFILES["standard"],
                    profile_name="standard",
                    block_size=block_size,
                )
            )
    # Async scheduling (ngram spec decode forces sync, so spec is None). The
    # preemption profile adds KV pressure so requests are preempted with
    # output still in flight.
    for layout in ("full", "hybrid"):
        for pc in (True, False):
            for profile_name in PROFILES:
                configs.append(
                    dict(
                        kv_layout=layout,
                        enable_prefix_caching=pc,
                        num_speculative_tokens=None,
                        profile=PROFILES[profile_name],
                        profile_name=profile_name,
                        async_scheduling=True,
                    )
                )
    return configs


SCHED_CONFIGS = _sched_configs()


def _config_id(config: dict[str, Any]) -> str:
    parts = [
        config["kv_layout"],
        config["profile_name"],
        f"pc{int(config['enable_prefix_caching'])}",
        f"spec{config['num_speculative_tokens']}",
    ]
    if config.get("block_size", 16) != 16:
        parts.append(f"bs{config['block_size']}")
    if config.get("stress_encoder_cache"):
        parts.append("tightenc")
    if config.get("async_scheduling"):
        parts.append("async")
    return "-".join(parts)


@pytest.mark.parametrize(
    "config", SCHED_CONFIGS, ids=[_config_id(c) for c in SCHED_CONFIGS]
)
def test_scheduler_statemachine(config: dict[str, Any]):
    machine_attrs = dict(config)
    machine_attrs.pop("profile_name")
    machine_cls = type(
        "BoundSchedulerFuzzStateMachine",
        (SchedulerFuzzStateMachine,),
        machine_attrs,
    )
    run_state_machine_as_test(machine_cls)
