# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark KV cache planning through the real V1 scheduler CPU path."""

import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.benchmark_kv_cache_manager import (
    get_kv_cache_block_sizes,
    make_kv_cache_config,
)
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


@dataclass(frozen=True)
class SchedulerBenchmarkResult:
    manager_backend: str
    cache_type: str
    cached_tokens_median: float
    admission_schedule_median_us: float
    finish_update_median_us: float
    decode_schedule_median_us: float
    decode_update_median_us: float


@dataclass(frozen=True)
class SchedulerBreakdownResult:
    manager_backend: str
    cache_type: str
    prefix_mode: str
    batch_size: int
    phase: str
    schedule_median_us: float
    schedule_p90_us: float
    kv_manager_median_us: float
    non_kv_median_us: float
    non_kv_share_median: float
    kv_calls_per_step_median: float


class _TimedKVManager:
    """Measure complete public KV-manager calls made by the scheduler."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._wrappers = {}
        self.reset()

    def reset(self) -> None:
        self.elapsed_ns = 0
        self.call_count = 0
        self.by_method_ns: dict[str, int] = defaultdict(int)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._manager, name)
        if not callable(value):
            return value
        wrapper = self._wrappers.get(name)
        if wrapper is None:

            def timed(*args: Any, **kwargs: Any) -> Any:
                start_ns = time.perf_counter_ns()
                try:
                    return getattr(self._manager, name)(*args, **kwargs)
                finally:
                    elapsed_ns = time.perf_counter_ns() - start_ns
                    self.elapsed_ns += elapsed_ns
                    self.call_count += 1
                    self.by_method_ns[name] += elapsed_ns

            wrapper = timed
            self._wrappers[name] = wrapper
        return wrapper


def _median_us(samples_ns: list[int]) -> float:
    return statistics.median(samples_ns) / 1_000


def _make_scheduler(
    *,
    cache_type: str,
    block_size: int,
    max_model_len: int,
    num_blocks: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    async_scheduling: bool = False,
) -> Scheduler:
    scheduler_block_size, hash_block_size, _ = get_kv_cache_block_sizes(
        cache_type, block_size
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
        hf_overrides={"max_position_embeddings": max_model_len},
    )
    cache_config = CacheConfig(
        block_size=scheduler_block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode="align" if cache_type == "hybrid-mamba" else "none",
    )
    cache_config.num_gpu_blocks = num_blocks
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        watermark=0.0,
        async_scheduling=async_scheduling,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
    )
    kv_cache_config: KVCacheConfig = make_kv_cache_config(
        cache_type, num_blocks, block_size, sliding_window=4096
    )
    register_all_kvcache_specs(vllm_config)
    scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    scheduler.use_v2_model_runner = False
    return scheduler


def _make_request(
    request_id: str,
    prompt_tokens: int,
    max_tokens: int,
    block_size: int,
    token_id: int = 0,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[token_id] * prompt_tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def _model_output(
    scheduler: Scheduler, scheduler_output: SchedulerOutput
) -> ModelRunnerOutput:
    request_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={
            request_id: index for index, request_id in enumerate(request_ids)
        },
        sampled_token_ids=[
            [] if scheduler.requests[request_id].is_prefill_chunk else [1]
            for request_id in request_ids
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _seed_prefix(
    scheduler: Scheduler,
    prompt_tokens: int,
    block_size: int,
    token_id: int = 0,
) -> None:
    producer = _make_request("producer", prompt_tokens, 1, block_size, token_id)
    scheduler.add_request(producer)
    while producer.request_id in scheduler.requests:
        scheduler_output = scheduler.schedule()
        scheduler.update_from_output(
            scheduler_output, _model_output(scheduler, scheduler_output)
        )


def run_scheduler_scenario(
    *,
    manager_backend: str,
    cache_type: str,
    prompt_tokens: int = 100_000,
    block_size: int = 16,
    max_model_len: int = 160_000,
    num_blocks: int = 50_000,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
    warmups: int = 5,
    iterations: int = 31,
    decode_steps: int = 301,
) -> SchedulerBenchmarkResult:
    """Measure scheduler admission, decode, and finish medians."""
    if manager_backend not in {"python", "rust"}:
        raise ValueError("manager_backend must be python or rust")
    if cache_type not in {"full", "hybrid-swa", "hybrid-mamba", "deepseek-v4"}:
        raise ValueError(
            "cache_type must be full, hybrid-swa, hybrid-mamba, or deepseek-v4"
        )
    previous_backend = os.environ.get("VLLM_USE_RUST_KV_CACHE_MANAGER")
    os.environ["VLLM_USE_RUST_KV_CACHE_MANAGER"] = (
        "1" if manager_backend == "rust" else "0"
    )
    try:
        _, hash_block_size, _ = get_kv_cache_block_sizes(cache_type, block_size)
        scheduler = _make_scheduler(
            cache_type=cache_type,
            block_size=block_size,
            max_model_len=max_model_len,
            num_blocks=num_blocks,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        _seed_prefix(scheduler, prompt_tokens, hash_block_size)

        admission_samples = []
        finish_samples = []
        hit_samples = []
        for index in range(warmups + iterations):
            request = _make_request(
                f"admission-{index}", prompt_tokens, 1, hash_block_size
            )
            scheduler.add_request(request)
            start_ns = time.perf_counter_ns()
            scheduler_output = scheduler.schedule()
            admission_ns = time.perf_counter_ns() - start_ns
            hit_tokens = scheduler_output.scheduled_new_reqs[0].num_computed_tokens
            start_ns = time.perf_counter_ns()
            scheduler.update_from_output(
                scheduler_output, _model_output(scheduler, scheduler_output)
            )
            finish_ns = time.perf_counter_ns() - start_ns
            if index >= warmups:
                admission_samples.append(admission_ns)
                finish_samples.append(finish_ns)
                hit_samples.append(hit_tokens)

        steady = _make_request(
            "steady", prompt_tokens, warmups + decode_steps + 1, hash_block_size
        )
        scheduler.add_request(steady)
        scheduler_output = scheduler.schedule()
        scheduler.update_from_output(
            scheduler_output, _model_output(scheduler, scheduler_output)
        )

        decode_samples = []
        update_samples = []
        for index in range(warmups + decode_steps):
            start_ns = time.perf_counter_ns()
            scheduler_output = scheduler.schedule()
            decode_ns = time.perf_counter_ns() - start_ns
            start_ns = time.perf_counter_ns()
            scheduler.update_from_output(
                scheduler_output, _model_output(scheduler, scheduler_output)
            )
            update_ns = time.perf_counter_ns() - start_ns
            if index >= warmups:
                decode_samples.append(decode_ns)
                update_samples.append(update_ns)

        return SchedulerBenchmarkResult(
            manager_backend=manager_backend,
            cache_type=cache_type,
            cached_tokens_median=statistics.median(hit_samples),
            admission_schedule_median_us=_median_us(admission_samples),
            finish_update_median_us=_median_us(finish_samples),
            decode_schedule_median_us=_median_us(decode_samples),
            decode_update_median_us=_median_us(update_samples),
        )
    finally:
        if previous_backend is None:
            os.environ.pop("VLLM_USE_RUST_KV_CACHE_MANAGER", None)
        else:
            os.environ["VLLM_USE_RUST_KV_CACHE_MANAGER"] = previous_backend


def _percentile_us(samples_ns: list[int], percentile: float) -> float:
    index = round((len(samples_ns) - 1) * percentile)
    return sorted(samples_ns)[index] / 1_000


def _make_profile_requests(
    prefix_mode: str,
    batch_size: int,
    iteration: int,
    prompt_tokens: int,
    output_tokens: int,
    block_size: int,
) -> list[Request]:
    return [
        _make_request(
            f"{prefix_mode}-{iteration}-{index}",
            prompt_tokens,
            output_tokens,
            block_size,
            token_id=(
                0 if prefix_mode == "shared" else iteration * batch_size + index + 1
            ),
        )
        for index in range(batch_size)
    ]


def _measure_schedule(
    scheduler: Scheduler,
    manager: _TimedKVManager,
) -> tuple[SchedulerOutput, int, int, int]:
    manager.reset()
    start_ns = time.perf_counter_ns()
    output = scheduler.schedule()
    total_ns = time.perf_counter_ns() - start_ns
    return output, total_ns, manager.elapsed_ns, manager.call_count


def run_scheduler_breakdown(
    *,
    manager_backend: str,
    cache_type: str,
    prefix_mode: str,
    batch_size: int,
    phase: str,
    prompt_tokens: int = 100_000,
    block_size: int = 16,
    max_model_len: int = 160_000,
    max_num_batched_tokens: int = 8192,
    warmups: int = 5,
    iterations: int = 31,
) -> SchedulerBreakdownResult:
    """Separate KV-manager time from the rest of async Scheduler.schedule."""
    if manager_backend not in {"python", "rust"}:
        raise ValueError("manager_backend must be python or rust")
    if cache_type not in {"full", "hybrid-swa", "hybrid-mamba", "deepseek-v4"}:
        raise ValueError(
            "cache_type must be full, hybrid-swa, hybrid-mamba, or deepseek-v4"
        )
    if prefix_mode not in {"shared", "independent"}:
        raise ValueError("prefix_mode must be shared or independent")
    if phase not in {"admission", "decode"}:
        raise ValueError("phase must be admission or decode")

    previous_backend = os.environ.get("VLLM_USE_RUST_KV_CACHE_MANAGER")
    os.environ["VLLM_USE_RUST_KV_CACHE_MANAGER"] = (
        "1" if manager_backend == "rust" else "0"
    )
    try:
        scheduler_block_size, hash_block_size, _ = get_kv_cache_block_sizes(
            cache_type, block_size
        )
        num_blocks = max(
            50_000,
            batch_size * prompt_tokens // scheduler_block_size + 10_000,
        )
        scheduler = _make_scheduler(
            cache_type=cache_type,
            block_size=block_size,
            max_model_len=max_model_len,
            num_blocks=num_blocks,
            max_num_seqs=batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            async_scheduling=True,
        )
        if prefix_mode == "shared":
            _seed_prefix(scheduler, prompt_tokens, hash_block_size)
        elif phase == "decode":
            for token_id in range(1, batch_size + 1):
                _seed_prefix(scheduler, prompt_tokens, hash_block_size, token_id)

        timed_manager = _TimedKVManager(scheduler.kv_cache_manager)
        scheduler.kv_cache_manager = timed_manager
        total_samples = []
        kv_samples = []
        call_samples = []

        if phase == "admission":
            for iteration in range(warmups + iterations):
                requests = _make_profile_requests(
                    prefix_mode,
                    batch_size,
                    iteration,
                    prompt_tokens,
                    100,
                    hash_block_size,
                )
                for request in requests:
                    scheduler.add_request(request)
                _, total_ns, kv_ns, call_count = _measure_schedule(
                    scheduler, timed_manager
                )
                scheduler.finish_requests(
                    [request.request_id for request in requests],
                    RequestStatus.FINISHED_ABORTED,
                )
                if iteration >= warmups:
                    total_samples.append(total_ns)
                    kv_samples.append(kv_ns)
                    call_samples.append(call_count)
        else:
            requests = _make_profile_requests(
                prefix_mode,
                batch_size,
                0,
                prompt_tokens,
                warmups + iterations + 16,
                hash_block_size,
            )
            for request in requests:
                scheduler.add_request(request)
            while any(request.num_output_tokens == 0 for request in requests):
                output = scheduler.schedule()
                scheduler.update_from_output(output, _model_output(scheduler, output))

            for iteration in range(warmups + iterations):
                output, total_ns, kv_ns, call_count = _measure_schedule(
                    scheduler, timed_manager
                )
                scheduler.update_from_output(output, _model_output(scheduler, output))
                if iteration >= warmups:
                    total_samples.append(total_ns)
                    kv_samples.append(kv_ns)
                    call_samples.append(call_count)

        non_kv_samples = [
            max(total_ns - kv_ns, 0)
            for total_ns, kv_ns in zip(total_samples, kv_samples)
        ]
        non_kv_shares = [
            non_kv_ns / total_ns
            for total_ns, non_kv_ns in zip(total_samples, non_kv_samples)
        ]
        return SchedulerBreakdownResult(
            manager_backend=manager_backend,
            cache_type=cache_type,
            prefix_mode=prefix_mode,
            batch_size=batch_size,
            phase=phase,
            schedule_median_us=_median_us(total_samples),
            schedule_p90_us=_percentile_us(total_samples, 0.9),
            kv_manager_median_us=_median_us(kv_samples),
            non_kv_median_us=_median_us(non_kv_samples),
            non_kv_share_median=statistics.median(non_kv_shares),
            kv_calls_per_step_median=statistics.median(call_samples),
        )
    finally:
        if previous_backend is None:
            os.environ.pop("VLLM_USE_RUST_KV_CACHE_MANAGER", None)
        else:
            os.environ["VLLM_USE_RUST_KV_CACHE_MANAGER"] = previous_backend


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark KV cache planning through Scheduler."
    )
    parser.add_argument(
        "--manager-backends",
        nargs="+",
        choices=("python", "rust"),
        default=["python", "rust"],
    )
    parser.add_argument(
        "--cache-type",
        choices=("full", "hybrid-swa", "hybrid-mamba", "deepseek-v4"),
        default="hybrid-mamba",
    )
    parser.add_argument("--prompt-tokens", type=int, default=100_000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=160_000)
    parser.add_argument("--num-blocks", type=int, default=50_000)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--decode-steps", type=int, default=301)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument(
        "--prefix-modes",
        nargs="+",
        choices=("shared", "independent"),
        default=["shared", "independent"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 32])
    parser.add_argument(
        "--phases", nargs="+", choices=("admission", "decode"), default=["decode"]
    )
    args = parser.parse_args()
    init_none_hash(sha256)
    if args.breakdown:
        results = [
            run_scheduler_breakdown(
                manager_backend=backend,
                cache_type=args.cache_type,
                prefix_mode=prefix_mode,
                batch_size=batch_size,
                phase=phase,
                prompt_tokens=args.prompt_tokens,
                block_size=args.block_size,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            for backend in args.manager_backends
            for prefix_mode in args.prefix_modes
            for batch_size in args.batch_sizes
            for phase in args.phases
        ]
    else:
        results = [
            run_scheduler_scenario(
                manager_backend=backend,
                cache_type=args.cache_type,
                prompt_tokens=args.prompt_tokens,
                block_size=args.block_size,
                max_model_len=args.max_model_len,
                num_blocks=args.num_blocks,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                warmups=args.warmups,
                iterations=args.iterations,
                decode_steps=args.decode_steps,
            )
            for backend in args.manager_backends
        ]
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    invoke_main()
