# Rust Hybrid KV Cache Manager

## Status

The opt-in Rust backend now implements a self-contained KV cache manager for a single full-attention group and for multi-group FullAttention plus Mamba configurations in `align` mode. This document describes the implemented ownership boundary, the long-prefix lookup optimization, the supported configuration, and the evidence required before enabling the backend more broadly.

The backend is selected with `VLLM_USE_RUST_KV_CACHE_MANAGER=1` when the scheduler is constructed. The Python manager remains the reference implementation, and unsupported configurations fail during construction instead of falling back to a partially native state machine.

## Motivation and measured result

Asynchronous scheduling can overlap KV cache planning with GPU execution, so a faster manager does not automatically improve end-to-end latency. Direct measurements through the real scheduler nevertheless show substantial CPU headroom for long cached prefixes. With a 100K-token cached prefix, 31 measured iterations, and medians rather than means, the retained implementation produced these admission results:

| Cache layout | Batch | Python median | Rust median | Change |
| --- | ---: | ---: | ---: | ---: |
| Full attention, block size 16 | 4 | 16.976 ms | 1.418 ms | -91.6% |
| Full attention, block size 16 | 32 | 130.972 ms | 10.707 ms | -91.8% |
| Qwen3.5 hybrid, block size 544 | 4 | 0.753 ms | 0.147 ms | -80.4% |
| Qwen3.5 hybrid, block size 544 | 32 | 5.705 ms | 1.044 ms | -81.7% |

End-to-end validation used Qwen3-4B on a GB200 with the Rust frontend, asynchronous scheduling, CUDA graphs, and no eager mode. The workload had 32 concurrent conversations sharing a 100K-token prefix and 20 turns that each added 100 input tokens and generated 100 output tokens. The median across two run-level results per backend improved TTFT by 17.2%, end-to-end latency by 4.1%, and throughput by 3.8%. Median TPOT changed by -0.1%, while mean, p90, and p99 TPOT also improved. Repeated TPOT measurements are required because scheduler work may be hidden by asynchronous GPU execution and a single run can be noisy.

## Goals

- Rust exclusively owns the mutable metadata for every supported cache group: the block arena, intrusive free/LRU queue, prefix-hash index, reference counts, parent links, request block tables, cached boundaries, and Mamba rolling-state bookkeeping.
- Python passes immutable request facts into high-level native operations and receives raw block IDs or token counts without callbacks from Rust into Python.
- Capacity checks spanning cache groups happen before mutation, so an unsuccessful allocation cannot partially update one group.
- The native backend preserves the observable scheduler contract for cache lookup, allocation, caching, eviction, reset, common-prefix queries, skipped-block release, and request release.
- Unsupported configurations fail early with an actionable error.

## Non-goals

- Rust does not own GPU KV payloads, launch kernels, tokenize prompts, compute request block hashes, or replace the scheduler in this change.
- This implementation does not add sliding-window, chunked-local, cross-attention, R-SWA, or sink-attention policies.
- It does not change eviction policy or cache-hit semantics.
- It does not add a scheduler-wide `plan_step` FFI call. The current high-level manager calls are retained because further call fusion did not demonstrate an incremental end-to-end gain.

## Ownership boundary

| Component | Owner | Responsibility |
| --- | --- | --- |
| Scheduler request state | Python | Token counts, request status, block hashes, scheduling policy, and model-facing output |
| Python KV adapter | Python | Configuration validation, conversion of request facts into native inputs, and wrapping raw block IDs |
| Block arena and free/LRU queue | Rust | Block identity, allocation generation, reference counts, cache metadata, eviction order, and free capacity |
| Full-attention policy | Rust | Prefix lookup, parent-path reconstruction, dense block-table growth, caching, and common-prefix counting |
| Hybrid coordinator | Rust | Cross-group hit reconciliation, capacity planning, allocation, Mamba state movement, skipped-block release, and request release |
| GPU model runner | Python/CUDA | Consumption of block tables and execution of the model |

The native manager is the only source of truth for mutable cache metadata. Python must not mirror reference counts, queue links, cache membership, parent links, or native request tables.

## Native data model

`BlockPool` stores a contiguous arena of block records and one intrusive free/LRU queue. Each record contains its reference count, optional group-qualified cache key, cached token boundary, allocation generation, optional full-attention parent reference, queue links, and queue membership. Block zero is the null block and is never placed in the free queue.

The cache index maps a block hash plus KV cache group ID to one or more block IDs. The group ID preserves independent cache residency for identical token prefixes in different groups while all groups share one physical allocation pool and eviction queue.

Each parent reference contains a block ID and the parent's generation. Allocating a physical block increments its generation. Path reconstruction validates the generation, cache group, cache-key presence, and expected length, preventing an evicted or reused parent from silently connecting a cached descendant to an unrelated allocation.

The full-attention manager stores a dense block table and cached boundary for each request. The hybrid manager stores one position-indexed table per group, a cached boundary per group, and Mamba `align` state for resident checkpoints. Mamba positions without a resident state use the null block ID, matching the Python contract.

`KVCacheBlockIds` keeps native results as `list[int]` values per cache group and exposes lazy `KVCacheBlock` views only when an existing scheduler path requests `.blocks`. This avoids allocating one Python object per cached block on the admission hot path while preserving the `KVCacheBlocks` interface.

## Operation semantics

### Full-attention prefix lookup

vLLM block hashes are cumulative. The native manager checks the first block and binary-searches the deepest cached cumulative hash, reducing Python hash extraction and hash-map probes from O(N) to O(log N) for an N-block candidate prefix. It then reconstructs the required O(N) block-ID table by walking parent links entirely in Rust.

If a parent path is absent or fails validation, the manager falls back to scalar forward lookup. This preserves the rule that a hit stops at the first missing block, including cases where a deeper cumulative hash remains cached after an ancestor was evicted.

### Hybrid lookup

Every full-attention group must contain the binary-search candidate, and every reconstructed parent path is validated independently. The Mamba policy then searches backward within the full-attention hit range for the newest reusable state checkpoint, truncates the full-attention tables to that position, and preserves the same-step Mamba reuse guard. Mamba blocks do not have parent links because a Mamba hit is one sparse state checkpoint rather than a dense prefix table.

### Allocation and cache commit

The manager computes evictable hit blocks and new physical blocks required by all groups, applies reserved-block and watermark constraints, and returns `None` before mutation when capacity is insufficient. On success it touches local hits before possible eviction, installs native request state, grows full-attention tables, advances Mamba state, and caches newly finalized blocks under group-qualified keys.

Computed block-ID vectors are moved into native request tables rather than cloned. Returned new allocations and cache-hit tables remain raw IDs until a compatibility consumer explicitly asks for block objects.

### Request release, eviction, and common prefix

Request release decrements group tables in reverse position order. Uncached blocks return to the free-queue head for immediate reuse, while cached blocks return to the tail to preserve LRU behavior. Reusing a cached free block removes its cache-index entry and parent metadata before allocation.

The full-attention common prefix is the leading run of blocks whose reference count equals the number of active native request tables. Mamba groups return zero because cascade attention does not consume Mamba checkpoints.

## Python interface and rejected FFI fusion

The adapter implements the existing `KVCacheManager` surface with high-level native calls. Each call completes one manager operation and exchanges immutable request facts, raw IDs, or scalar results. Python performs scheduler policy and validation but does not participate in an in-progress native mutation.

An experiment kept cache-hit IDs as pending Rust state between lookup and allocation. It reduced the 32-request full-attention admission median from about 11.0 ms to 4.8 ms, but repeated end-to-end measurements showed no incremental benefit because asynchronous scheduling hid the remaining CPU work. It also introduced Python-visible pending state and additional cleanup paths, so it was reverted. A future `plan_step` or pending-admission interface must first demonstrate a stable end-to-end gain on a workload that crosses the scheduler-overlap threshold.

## Supported configuration

- One `FullAttentionSpec` group, or multiple groups containing only `FullAttentionSpec` and `MambaSpec` with at least one group of each type.
- Every Mamba group uses `mamba_cache_mode="align"` and has no speculative blocks.
- Cache, scheduler, and hash block sizes are identical, with DCP/PCP world size one.
- No EAGLE/MTP, KV connector, KV cache event publisher, KV cache metrics collector, deferred free, external computed KV, or encoder KV allocation.
- The single-group full-attention backend does not support KV cache zeroing. The hybrid adapter exposes the existing block-zeroing bookkeeping required by Mamba align mode.

## Rust module layout

- `block_pool.rs` owns block records, generations, parent references, the intrusive free/LRU queue, group-qualified cache indices, allocation, touching, eviction, and release.
- `full_attention.rs` owns the single-group request registry, prefix lookup, dense table growth, caching, and common-prefix logic.
- `hybrid.rs` owns multi-group request state, FullAttention/Mamba hit reconciliation, capacity planning, allocation ordering, Mamba transitions, zeroing block IDs, and public hybrid methods.
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

## Validation

Focused pytest coverage compares cache hits, allocation, eviction, reset, an evicted parent with a cached descendant, hybrid group routing, Mamba state lifecycle, and lazy block-ID compatibility. CPU performance tests use medians over 31 measured iterations and require the Rust implementation to be faster for the covered long-context manager operations and real scheduler paths.

```bash
.venv/bin/python -m pytest tests/v1/core/test_prefix_caching.py -q -k 'rust_'
.venv/bin/python -m benchmarks.benchmark_scheduler_kv_cache --breakdown --cache-type full --prefix-modes shared --batch-sizes 4 32 --phases admission
.venv/bin/python -m benchmarks.benchmark_scheduler_kv_cache --breakdown --cache-type hybrid-mamba --prefix-modes shared --batch-sizes 4 32 --phases admission --block-size 544
```

Model-level validation uses the repository Rust `vllm-bench`, the Rust frontend, default asynchronous scheduling, CUDA graphs, and no `--enforce-eager`. TTFT is compared at matching turn indices or as a run-level distribution rather than averaging turns whose context lengths differ. TPOT p50, mean, and tail percentiles must not regress across repeated A/B runs.

## Rollout and risks

The backend remains opt-in while the Python implementation is the reference. The highest correctness risk is stale parent metadata after eviction or block reuse, which is addressed by cache-key validation, allocation generations, scalar fallback, and focused tests. The highest performance risk is that asynchronous scheduling hides CPU savings; therefore both scheduler-only medians and repeated model-level TTFT, TPOT, latency, and throughput results are required before expanding support.
