# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark KV cache planning through the real V1 scheduler CPU path."""

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

from benchmarks.benchmark_kv_cache_manager import make_kv_cache_config
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
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
) -> Scheduler:
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
        block_size=block_size,
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
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        hash_block_size=block_size,
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
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[0] * prompt_tokens,
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


def _seed_prefix(scheduler: Scheduler, prompt_tokens: int, block_size: int) -> None:
    producer = _make_request("producer", prompt_tokens, 1, block_size)
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
    if cache_type not in {"full", "hybrid-mamba"}:
        raise ValueError("cache_type must be full or hybrid-mamba")
    previous_backend = os.environ.get("VLLM_USE_RUST_KV_CACHE_MANAGER")
    os.environ["VLLM_USE_RUST_KV_CACHE_MANAGER"] = (
        "1" if manager_backend == "rust" else "0"
    )
    try:
        scheduler = _make_scheduler(
            cache_type=cache_type,
            block_size=block_size,
            max_model_len=max_model_len,
            num_blocks=num_blocks,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        _seed_prefix(scheduler, prompt_tokens, block_size)

        admission_samples = []
        finish_samples = []
        hit_samples = []
        for index in range(warmups + iterations):
            request = _make_request(f"admission-{index}", prompt_tokens, 1, block_size)
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
            "steady", prompt_tokens, warmups + decode_steps + 1, block_size
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
        "--cache-type", choices=("full", "hybrid-mamba"), default="hybrid-mamba"
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
    args = parser.parse_args()
    init_none_hash(sha256)
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
