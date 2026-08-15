# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark CPU metadata operations in ``KVCacheManager``.

The benchmark intentionally uses the public manager API. Request construction,
block hashing, and scenario setup happen outside the timed regions so the
results isolate prefix lookup, slot allocation, and request block release.
"""

import argparse
import gc
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from tabulate import tabulate

from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


@dataclass(frozen=True)
class BenchmarkResult:
    manager_backend: str
    cache_type: str
    prompt_tokens: int
    cached_tokens: int
    hit_rate: float
    operation: str
    median_us: float
    p95_us: float


def _percentile(samples: list[int], percentile: float) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    rank = math.ceil(percentile * len(samples)) - 1
    return float(sorted(samples)[max(0, rank)])


def _timing_result(
    samples_ns: list[int],
    *,
    manager_backend: str,
    cache_type: str,
    prompt_tokens: int,
    cached_tokens: int,
    operation: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        manager_backend=manager_backend,
        cache_type=cache_type,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        hit_rate=cached_tokens / prompt_tokens,
        operation=operation,
        median_us=statistics.median(samples_ns) / 1_000,
        p95_us=_percentile(samples_ns, 0.95) / 1_000,
    )


def _full_attention_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _sliding_window_spec(block_size: int, sliding_window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )


def _mamba_spec(block_size: int) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
    )


def make_kv_cache_config(
    cache_type: str,
    num_blocks: int,
    block_size: int,
    sliding_window: int,
) -> KVCacheConfig:
    specs = {
        "full": [_full_attention_spec(block_size)],
        "hybrid-swa": [
            _full_attention_spec(block_size),
            _sliding_window_spec(block_size, sliding_window),
        ],
        "hybrid-mamba": [
            _full_attention_spec(block_size),
            _mamba_spec(block_size),
        ],
        "hybrid-all": [
            _full_attention_spec(block_size),
            _sliding_window_spec(block_size, sliding_window),
            _mamba_spec(block_size),
        ],
    }[cache_type]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{group_id}"], spec)
            for group_id, spec in enumerate(specs)
        ],
    )


def _make_request(
    request_id: str,
    token_ids: list[int],
    block_size: int,
    hash_fn: Callable,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )


def _time_ns(function: Callable[[], object]) -> tuple[int, object]:
    start_ns = time.perf_counter_ns()
    result = function()
    return time.perf_counter_ns() - start_ns, result


def _run_allocation_cycle(
    manager: Any,
    request: Request,
    expected_cached_tokens: int,
) -> tuple[int, int, int]:
    manager.new_step_starts()
    lookup_ns, lookup_result = _time_ns(lambda: manager.get_computed_blocks(request))
    cached_blocks, cached_tokens, _ = lookup_result
    assert isinstance(cached_blocks, KVCacheBlocks)
    if cached_tokens != expected_cached_tokens:
        raise RuntimeError(
            "prefix-cache scenario changed while benchmarking: "
            f"expected {expected_cached_tokens} cached tokens, got {cached_tokens}"
        )

    allocate_ns, allocated_blocks = _time_ns(
        lambda: manager.allocate_slots(
            request,
            num_new_tokens=request.num_tokens - cached_tokens,
            num_new_computed_tokens=cached_tokens,
            new_computed_blocks=cached_blocks,
            delay_cache_blocks=True,
        )
    )
    if allocated_blocks is None:
        raise RuntimeError("KVCacheManager ran out of blocks in benchmark setup")
    free_ns, _ = _time_ns(lambda: manager.free(request))
    return lookup_ns, allocate_ns, free_ns


def run_scenario(
    *,
    manager_backend: str,
    cache_type: str,
    prompt_tokens: int,
    hit_rate: float,
    block_size: int,
    sliding_window: int,
    warmups: int,
    iterations: int,
) -> list[BenchmarkResult]:
    max_cached_blocks = (prompt_tokens - 1) // block_size
    requested_cached_blocks = int(max_cached_blocks * hit_rate)
    requested_cached_tokens = requested_cached_blocks * block_size

    num_groups = 1 + cache_type.count("hybrid")
    if cache_type == "hybrid-all":
        num_groups = 3
    prompt_blocks = math.ceil(prompt_tokens / block_size)
    num_blocks = (num_groups + 1) * (prompt_blocks + 2) + 64
    config = make_kv_cache_config(cache_type, num_blocks, block_size, sliding_window)
    manager_kwargs = dict(
        kv_cache_config=config,
        max_model_len=prompt_tokens,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
    )
    if manager_backend == "python":
        manager = KVCacheManager(**manager_kwargs)
    else:
        from vllm.v1.core.rust_kv_cache_manager import (
            RustFullAttentionKVCacheManager,
        )

        manager = RustFullAttentionKVCacheManager(**manager_kwargs)

    token_ids = [token_id % 32_000 for token_id in range(prompt_tokens)]
    if requested_cached_tokens:
        producer = _make_request(
            "producer",
            token_ids[:requested_cached_tokens],
            block_size,
            sha256,
        )
        producer_blocks = manager.allocate_slots(
            producer, num_new_tokens=producer.num_tokens
        )
        if producer_blocks is None:
            raise RuntimeError("failed to populate the prefix cache")
        manager.free(producer)

    consumer = _make_request("consumer", token_ids, block_size, sha256)
    manager.new_step_starts()
    _, expected_cached_tokens, _ = manager.get_computed_blocks(consumer)
    if expected_cached_tokens != requested_cached_tokens:
        raise RuntimeError(
            f"{cache_type} cached {expected_cached_tokens} tokens; "
            f"the scenario requested {requested_cached_tokens}"
        )

    for _ in range(warmups):
        _run_allocation_cycle(manager, consumer, expected_cached_tokens)

    lookup_samples: list[int] = []
    allocate_samples: list[int] = []
    free_samples: list[int] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(iterations):
            lookup_ns, allocate_ns, free_ns = _run_allocation_cycle(
                manager, consumer, expected_cached_tokens
            )
            lookup_samples.append(lookup_ns)
            allocate_samples.append(allocate_ns)
            free_samples.append(free_ns)
    finally:
        gc.enable()

    steady_request = _make_request("steady", token_ids, block_size, sha256)
    if manager.allocate_slots(steady_request, steady_request.num_tokens) is None:
        raise RuntimeError("failed to prepare steady-state allocation benchmark")
    steady_request.num_computed_tokens = steady_request.num_tokens - 1

    for _ in range(warmups):
        manager.allocate_slots(steady_request, 1)
        manager.get_num_common_prefix_blocks(steady_request.request_id)

    steady_allocate_samples: list[int] = []
    common_prefix_samples: list[int] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(iterations):
            elapsed_ns, new_blocks = _time_ns(
                lambda: manager.allocate_slots(steady_request, 1)
            )
            if new_blocks is None:
                raise RuntimeError("steady-state slot allocation unexpectedly failed")
            steady_allocate_samples.append(elapsed_ns)
            elapsed_ns, _ = _time_ns(
                lambda: manager.get_num_common_prefix_blocks(steady_request.request_id)
            )
            common_prefix_samples.append(elapsed_ns)
    finally:
        gc.enable()
        manager.free(steady_request)

    common = {
        "manager_backend": manager_backend,
        "cache_type": cache_type,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": expected_cached_tokens,
    }
    return [
        _timing_result(lookup_samples, operation="lookup", **common),
        _timing_result(allocate_samples, operation="allocate", **common),
        _timing_result(free_samples, operation="free", **common),
        _timing_result(steady_allocate_samples, operation="steady_allocate", **common),
        _timing_result(common_prefix_samples, operation="common_prefix", **common),
    ]


def _validate_args(args: argparse.Namespace) -> None:
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.sliding_window <= 0:
        raise ValueError("--sliding-window must be positive")
    if args.sliding_window % args.block_size:
        raise ValueError("--sliding-window must be a multiple of --block-size")
    if args.warmups < 0 or args.iterations <= 0:
        raise ValueError("--warmups must be non-negative and --iterations positive")
    if any(length <= 1 for length in args.prompt_lengths):
        raise ValueError("--prompt-lengths must contain values greater than one")
    if any(not 0 <= hit_rate <= 1 for hit_rate in args.hit_rates):
        raise ValueError("--hit-rates values must be between zero and one")
    if "rust" in args.manager_backends and args.cache_types != ["full"]:
        raise ValueError("the Rust manager benchmark supports --cache-types full only")
    if args.assert_rust_faster and set(args.manager_backends) != {"python", "rust"}:
        raise ValueError("--assert-rust-faster requires --manager-backends python rust")


def _assert_rust_faster(results: list[BenchmarkResult]) -> None:
    by_key = {
        (
            result.manager_backend,
            result.cache_type,
            result.prompt_tokens,
            result.cached_tokens,
            result.operation,
        ): result
        for result in results
    }
    failures = []
    for key, python_result in by_key.items():
        backend, *scenario = key
        if backend != "python":
            continue
        rust_result = by_key[("rust", *scenario)]
        if rust_result.median_us >= python_result.median_us:
            failures.append(
                f"{scenario}: Rust {rust_result.median_us:.3f} us >= "
                f"Python {python_result.median_us:.3f} us"
            )
    if failures:
        raise RuntimeError("Rust performance gate failed:\n" + "\n".join(failures))


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    init_none_hash(sha256)
    results = [
        result
        for manager_backend in args.manager_backends
        for cache_type in args.cache_types
        for prompt_tokens in args.prompt_lengths
        for hit_rate in args.hit_rates
        for result in run_scenario(
            manager_backend=manager_backend,
            cache_type=cache_type,
            prompt_tokens=prompt_tokens,
            hit_rate=hit_rate,
            block_size=args.block_size,
            sliding_window=args.sliding_window,
            warmups=args.warmups,
            iterations=args.iterations,
        )
    ]

    if args.assert_rust_faster:
        _assert_rust_faster(results)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

    print(
        tabulate(
            [
                [
                    result.manager_backend,
                    result.cache_type,
                    result.prompt_tokens,
                    result.cached_tokens,
                    result.hit_rate,
                    result.operation,
                    result.median_us,
                    result.p95_us,
                ]
                for result in results
            ],
            headers=[
                "Manager",
                "Cache type",
                "Prompt tokens",
                "Cached tokens",
                "Hit rate",
                "Operation",
                "Median (us)",
                "P95 (us)",
            ],
            tablefmt="grid",
            floatfmt=("", "", "", ".3f", "", ".3f", ".3f"),
        )
    )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark KVCacheManager CPU metadata operations."
    )
    parser.add_argument(
        "--manager-backends",
        nargs="+",
        choices=("python", "rust"),
        default=["python"],
    )
    parser.add_argument(
        "--cache-types",
        nargs="+",
        choices=("full", "hybrid-swa", "hybrid-mamba", "hybrid-all"),
        default=["full", "hybrid-swa", "hybrid-mamba", "hybrid-all"],
    )
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", default=[4096, 32768, 131072]
    )
    parser.add_argument("--hit-rates", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sliding-window", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--assert-rust-faster",
        action="store_true",
        help="Fail if any matched Rust operation has a slower median than Python.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable results."
    )
    main(parser.parse_args())


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
