# Rust Hybrid KV Cache Manager

## Status

The opt-in Rust backend now implements a self-contained KV cache manager for a single full-attention group, heterogeneous FullAttention plus SlidingWindow groups, and FullAttention plus Mamba configurations in `align` mode. This document describes the implemented ownership boundary, the long-prefix lookup optimization, the supported configuration, and the evidence required before enabling the backend more broadly.

The backend is selected with `VLLM_USE_RUST_KV_CACHE_MANAGER=1` when the scheduler is constructed. The Python manager remains the reference implementation, and unsupported configurations fail during construction instead of falling back to a partially native state machine.

## Motivation and measured result

Asynchronous scheduling can overlap KV cache planning with GPU execution, so a faster manager does not automatically improve end-to-end latency. Direct measurements through the real scheduler nevertheless show substantial CPU headroom for long cached prefixes. With a 100K-token cached prefix, 31 measured iterations, and medians rather than means, the retained implementation produced these admission results:

| Cache layout | Batch | Python median | Rust median | Change |
| --- | ---: | ---: | ---: | ---: |
| Full attention, block size 16 | 4 | 16.976 ms | 1.418 ms | -91.6% |
| Full attention, block size 16 | 32 | 130.972 ms | 10.707 ms | -91.8% |
| Qwen3.5 hybrid, block size 544 | 4 | 0.753 ms | 0.147 ms | -80.4% |
| Qwen3.5 hybrid, block size 544 | 32 | 5.705 ms | 1.044 ms | -81.7% |
| DeepSeek-V4 layout, admission | 4 | 5.002 ms | 1.817 ms | -63.7% |
| DeepSeek-V4 layout, admission | 32 | 39.465 ms | 39.593 ms | +0.3% |
| DeepSeek-V4 layout, decode | 4 | 0.085 ms | 0.038 ms | -55.7% |
| DeepSeek-V4 layout, decode | 32 | 0.499 ms | 0.215 ms | -56.9% |

The DeepSeek-V4 rows use its five-group cache geometry: one full MLA group with block size 256, two sliding-window MLA groups with block size 64 and window 128, one sliding-window MLA group with block size 4 and window 8, and one sliding-window MLA group with block size 8 and window 128. The scheduler block size is 256 and the hash block size is 4. They exercise the real asynchronous scheduler CPU path with synthetic requests, not a model forward pass. A focused repeat of the batch-32 admission case measured 39.353 ms for Python and 39.986 ms for Rust, confirming that this high-admission-concurrency point has no scheduler-level speedup despite the direct manager operations being faster.

End-to-end validation used Qwen3-4B on a GB200 with the Rust frontend, asynchronous scheduling, CUDA graphs, and no eager mode. The workload had 32 concurrent conversations sharing a 100K-token prefix and 20 turns that each added 100 input tokens and generated 100 output tokens. The median across two run-level results per backend improved TTFT by 17.2%, end-to-end latency by 4.1%, and throughput by 3.8%. Median TPOT changed by -0.1%, while mean, p90, and p99 TPOT also improved. Repeated TPOT measurements are required because scheduler work may be hidden by asynchronous GPU execution and a single run can be noisy.

DeepSeek-V4-Flash-0731 with DSpark was also validated on a GB200 at batch size one with the Rust frontend, asynchronous scheduling, CUDA graphs, a 100K-token initial context, and 10 warmup plus 10 measured turns that each added 100 input tokens and generated 100 output tokens. The context-aware medians improved TTFT by 3.90%, TPOT by 4.19%, inter-token latency by 0.31%, and end-to-end latency by 4.41%. DSpark acceptance length differed between the two single runs, so the TPOT result establishes that there was no observed regression but does not isolate the KV manager's causal contribution.

## Goals

- Rust exclusively owns the mutable metadata for every supported cache group: the block arena, intrusive free/LRU queue, prefix-hash index, reference counts, parent links, request block tables, cached boundaries, and Mamba rolling-state bookkeeping.
- Python passes immutable request facts into high-level native operations and receives raw block IDs or token counts without callbacks from Rust into Python.
- Capacity checks spanning cache groups happen before mutation, so an unsuccessful allocation cannot partially update one group.
- The native backend preserves the observable scheduler contract for cache lookup, allocation, caching, eviction, reset, common-prefix queries, skipped-block release, and request release.
- Unsupported configurations fail early with an actionable error.

## Non-goals

- Rust does not own GPU KV payloads, launch kernels, tokenize prompts, compute request block hashes, or replace the scheduler in this change.
- This implementation does not add chunked-local, cross-attention, R-SWA, or sink-attention policies.
- It does not change eviction policy or cache-hit semantics.
- It does not add a scheduler-wide `plan_step` FFI call. The current high-level manager calls are retained because further call fusion did not demonstrate an incremental end-to-end gain.

## Ownership boundary

| Component | Owner | Responsibility |
| --- | --- | --- |
| Scheduler request state | Python | Token counts, request status, block hashes, scheduling policy, and model-facing output |
| Python KV adapter | Python | Configuration validation, conversion of request facts into native inputs, and wrapping raw block IDs |
| Block arena and free/LRU queue | Rust | Block identity, allocation generation, reference counts, cache metadata, eviction order, and free capacity |
| Full-attention policy | Rust | Prefix lookup, parent-path reconstruction, dense block-table growth, caching, and common-prefix counting |
| Hybrid coordinator | Rust | Cross-group hit reconciliation, capacity planning, allocation, sliding-window recycling, Mamba state movement, skipped-block release, and request release |
| GPU model runner | Python/CUDA | Consumption of block tables and execution of the model |

The native manager is the only source of truth for mutable cache metadata. Python must not mirror reference counts, queue links, cache membership, parent links, or native request tables.

## Native data model

`BlockPool` stores a contiguous arena of block records and one intrusive free/LRU queue. Each record contains its reference count, optional group-qualified cache key, cached token boundary, allocation generation, optional full-attention parent reference, queue links, and queue membership. Block zero is the null block and is never placed in the free queue.

The cache index maps a block hash plus KV cache group ID to one or more block IDs. The group ID preserves independent cache residency for identical token prefixes in different groups while all groups share one physical allocation pool and eviction queue.

Each parent reference contains a block ID and the parent's generation. Allocating a physical block increments its generation. Path reconstruction validates the generation, cache group, cache-key presence, and expected length, preventing an evicted or reused parent from silently connecting a cached descendant to an unrelated allocation.

The full-attention manager stores a dense block table and cached boundary for each request. The hybrid manager stores one position-indexed table per group, a cached boundary per group, a released-prefix cursor for sliding-window groups, and Mamba `align` state for resident checkpoints. Skipped sliding-window and Mamba positions use the null block ID, matching the Python contract.

`KVCacheBlockIds` keeps native results as `list[int]` values per cache group and exposes lazy `KVCacheBlock` views only when an existing scheduler path requests `.blocks`. The hybrid adapter marks an unchanged native lookup result so allocation can consume the matching request state directly in Rust instead of converting a large sparse block table back through FFI. This avoids allocating one Python object per cached block and avoids the Rust-to-Python-to-Rust round trip on the admission hot path while preserving the `KVCacheBlocks` interface.

## Operation semantics

### Full-attention prefix lookup

vLLM block hashes are cumulative. The native manager checks the first block and binary-searches the deepest cached cumulative hash, reducing Python hash extraction and hash-map probes from O(N) to O(log N) for an N-block candidate prefix. It then reconstructs the required O(N) block-ID table by walking parent links entirely in Rust.

If a parent path is absent or fails validation, the manager falls back to scalar forward lookup. This preserves the rule that a hit stops at the first missing block, including cases where a deeper cumulative hash remains cached after an ancestor was evicted.

### Hybrid lookup

Every full-attention group must contain the binary-search candidate, and every reconstructed parent path is validated independently. Each sliding-window group searches backward for the latest scheduler-aligned boundary with the required contiguous window tail, materializes null IDs for skipped positions, and participates in a fixed-point reconciliation so every group agrees on the final hit length. The Mamba policy searches backward within the full-attention hit range for the newest reusable state checkpoint, truncates the full-attention tables to that position, and preserves the same-step Mamba reuse guard. Sliding-window tails and Mamba checkpoints do not have full-attention parent chains.

### Allocation and cache commit

The manager computes evictable hit blocks and new physical blocks required by all groups, applies sliding-window admission caps, reserved-block constraints, and watermark constraints, and returns `None` before mutation when capacity is insufficient. On success it touches local hits before possible eviction, installs native request state, grows attention tables, advances Mamba state, and caches newly finalized blocks under group-qualified keys. Sliding-window release uses a monotonic per-request cursor so decode only examines newly skipped blocks instead of rescanning the entire sparse prefix on every step.

An unchanged lookup result remains pending in Rust until allocation in the same scheduler step. Allocation moves those computed block-ID vectors into native request tables without extracting the Python lists. `new_step_starts`, request release, reset, and explicit block-table truncation discard stale pending results. Returned new allocations and cache-hit tables remain raw IDs until a compatibility consumer explicitly asks for block objects.

### Request release, eviction, and common prefix

Request release decrements group tables in reverse position order. Uncached blocks return to the free-queue head for immediate reuse, while cached blocks return to the tail to preserve LRU behavior. Reusing a cached free block removes its cache-index entry and parent metadata before allocation.

The full-attention common prefix is the leading run of blocks whose reference count equals the number of active native request tables. Mamba groups return zero because cascade attention does not consume Mamba checkpoints.

## Python interface and FFI continuity

The adapter implements the existing `KVCacheManager` surface with high-level native calls. Each call completes one manager operation and exchanges immutable request facts, raw IDs, or scalar results. Python performs scheduler policy and validation but does not participate in an in-progress native mutation.

The adapter preserves native continuity between prefix lookup and allocation. Python still receives block IDs for the model-facing scheduler contract, but an internal marker allows allocation to consume the corresponding pending native vectors without sending every ID back through FFI. This matters for DeepSeek-V4: a 100K hit contains about 41K position entries across block sizes 256, 64, 4, and 8. With the native continuity path, the Rust allocation median was 64.3 microseconds versus 318.9 microseconds for Python, while Rust lookup was 156.1 microseconds versus 638.9 microseconds for Python.

A future scheduler-wide `plan_step` interface could fuse multiple requests into one call, but it remains out of scope until it demonstrates additional end-to-end value beyond asynchronous scheduling and the current per-request native continuity.

## Supported configuration

- One `FullAttentionSpec` group, multiple groups containing FullAttention and SlidingWindow specs, or FullAttention plus Mamba specs.
- FullAttention and SlidingWindow groups may use different block sizes when every group block size divides the scheduler block size and is divisible by the hash block size. `MLAAttentionSpec` and `SlidingWindowMLASpec` use these same policies.
- Every Mamba group uses `mamba_cache_mode="align"`, has no speculative blocks, and uses the same block size as the scheduler and hash granularity. Mamba and SlidingWindow groups cannot currently be combined.
- The covered important layouts are gpt-oss FullAttention plus SlidingWindow, base Gemma 4 FullAttention plus SlidingWindow with `extra_retained_tokens`, DeepSeek-V4 Full MLA plus two SlidingWindow MLA groups with mixed block sizes, and GLM-5.2 full MLA.
- DCP/PCP world size must be one. R-SWA, sink attention, and chunked-local attention are rejected.
- Single-lookahead EAGLE/DSpark is supported on scheduler-sized FullAttention and SlidingWindow groups, including the DeepSeek-V4 DSpark layout. Multi-module MTP, KV connector, KV cache event publisher, KV cache metrics collector, deferred free, external computed KV, and encoder KV allocation are not supported.
- The single-group full-attention backend does not support KV cache zeroing. The hybrid adapter exposes block-zeroing bookkeeping required by Mamba align and DeepSeek-V4 cache layouts.

## Rust module layout

- `block_pool.rs` owns block records, generations, parent references, the intrusive free/LRU queue, group-qualified cache indices, allocation, touching, eviction, and release.
- `full_attention.rs` owns the single-group request registry, prefix lookup, dense table growth, caching, and common-prefix logic.
- `hybrid.rs` owns multi-group request state, FullAttention/SlidingWindow/Mamba hit reconciliation, capacity planning, allocation ordering, sliding-window recycling, Mamba transitions, pending native hits, zeroing block IDs, and public hybrid methods.
- `lib.rs` registers the PyO3 module and exports the native manager classes.
- `vllm/v1/core/rust_kv_cache_manager.py` validates configuration and adapts scheduler calls without owning native cache metadata.

## Correctness invariants

- Every non-null block is either referenced and absent from the free queue, or unreferenced and present exactly once in the free queue.
- Every cached block has a matching group-qualified cache-index entry, and eviction removes the index entry before reuse.
- Every valid full-attention parent reference points to the expected cache group and allocation generation.
- A request table never references an unallocated block except for the distinguished null block in sparse Mamba positions.
- A failed cross-group capacity check leaves block metadata, request tables, cache indices, and queue order unchanged.
- Local hit blocks for every group are touched before allocating a block that may evict cache entries.
- The reconciled hybrid hit never exceeds any individual group hit and remains scheduler-block aligned.
- A sliding-window release cursor never moves backward, and already released null positions are not rescanned on later decode steps.
- A pending native hit is consumed only by the matching request, and every step boundary or explicit truncation invalidates unused pending state.

## Validation

Focused pytest coverage compares cache hits, allocation, eviction, reset, an evicted parent with a cached descendant, hybrid group routing, Mamba state lifecycle, and lazy block-ID compatibility. CPU performance tests use medians over 31 measured iterations and require the Rust implementation to be faster for the covered long-context manager operations and real scheduler paths.

```bash
.venv/bin/python -m pytest tests/v1/core/test_prefix_caching.py -q -k 'rust_'
.venv/bin/python -m benchmarks.benchmark_scheduler_kv_cache --breakdown --cache-type full --prefix-modes shared --batch-sizes 4 32 --phases admission
.venv/bin/python -m benchmarks.benchmark_scheduler_kv_cache --breakdown --cache-type hybrid-mamba --prefix-modes shared --batch-sizes 4 32 --phases admission --block-size 544
.venv/bin/python -m benchmarks.benchmark_kv_cache_manager --manager-backends python rust --cache-types deepseek-v4 --prompt-lengths 100000 --hit-rates 1.0 --warmups 5 --iterations 31 --assert-rust-faster
.venv/bin/python -m benchmarks.benchmark_scheduler_kv_cache --breakdown --cache-type deepseek-v4 --prefix-modes shared --batch-sizes 4 32 --phases admission decode --prompt-tokens 100000 --warmups 5 --iterations 31
```

Model-level validation uses the repository Rust `vllm-bench`, the Rust frontend, default asynchronous scheduling, CUDA graphs, and no `--enforce-eager`. TTFT is compared at matching turn indices or as a run-level distribution rather than averaging turns whose context lengths differ. TPOT p50, mean, and tail percentiles must not regress across repeated A/B runs.

## Rollout and risks

The backend remains opt-in while the Python implementation is the reference. The highest correctness risk is stale parent metadata after eviction or block reuse, which is addressed by cache-key validation, allocation generations, scalar fallback, and focused tests. The highest performance risk is that asynchronous scheduling hides CPU savings; therefore both scheduler-only medians and repeated model-level TTFT, TPOT, latency, and throughput results are required before expanding support.
