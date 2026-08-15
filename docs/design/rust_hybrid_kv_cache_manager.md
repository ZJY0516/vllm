# Rust Hybrid KV Cache Manager

## Status

This document describes the implementation plan for a self-contained Rust KV cache manager for vLLM V1. The first hybrid target is one full-attention group plus one Mamba group in `align` mode, matching the cache structure used by Qwen3.5. Sliding-window support follows as another native group policy after the shared hybrid core is established.

## Motivation

The Python KV cache manager is not the end-to-end bottleneck in every workload because asynchronous scheduling can overlap CPU planning with GPU execution, but direct scheduler measurements show that the metadata path has substantial headroom. A CPU-only `Scheduler` test using a 100K-token initial context and a 200-turn conversation, with every turn adding 100 input tokens and generating 100 output tokens, produced the following medians.

| Manager | Admission schedule | Decode schedule step | Finish update | Total schedule time per turn |
| --- | ---: | ---: | ---: | ---: |
| Python full attention | 5.872 ms | 294.3 us | 644.0 us | 34.926 ms |
| Python full attention + sliding window | 6.112 ms | 295.0 us | 941.0 us | 35.628 ms |
| Python full attention + Mamba | 6.518 ms | 296.5 us | 948.8 us | 36.569 ms |
| Rust full attention | 1.534 ms | 16.4 us | 51.1 us | 3.285 ms |

The final 25 turns increase the Python full-attention schedule total to 39.811 ms, the Python Mamba hybrid total to 41.513 ms, and the Rust full-attention total to 3.638 ms. The dominant repeated cost is the full-attention block-table and common-prefix work shared by both hybrid layouts, while Mamba adds the largest admission and release overhead. Rewriting only isolated Mamba or sliding-window functions would leave the shared coordinator, block pool, request tables, and cross-group transitions in Python and would not capture most of the available scheduler reduction.

## Goals

- Rust exclusively owns mutable KV cache metadata for every supported group, including the block arena, intrusive free/LRU queue, prefix-hash indices, reference counts, request block tables, cached boundaries, Mamba rolling-state bookkeeping, and pending block-copy actions.
- The scheduler selects one Python adapter at construction time, and the adapter delegates high-level cache operations to one native state machine without calling back into Python during a transition.
- The first hybrid implementation preserves the observable behavior of `KVCacheManager`, `HybridKVCacheCoordinator`, `FullAttentionManager`, and `MambaManager` for a two-group full-attention plus Mamba `align` configuration.
- Capacity checks that span groups are performed before mutation so an unsuccessful allocation cannot partially update one group.
- Unsupported configurations fail during scheduler construction with an actionable error instead of silently falling back to partially native behavior.
- Correctness is established through public scheduler and manager APIs, and CPU performance is guarded with medians over long-context operations.

## Non-goals

- Rust does not own GPU KV payloads, launch kernels, tokenize prompts, or compute request block hashes in the first implementation.
- The first hybrid implementation does not support KV connectors, KV events, KV metrics collection, cache zeroing, EAGLE/MTP, DCP, PCP, speculative Mamba blocks, or different hash and scheduler block sizes.
- The first hybrid implementation does not change the existing eviction policy or cache-hit semantics.
- Sliding-window, chunked-local, cross-attention, R-SWA, and sink-attention policies are not part of the first code change, although the native group-policy boundary must allow them to be added without duplicating the block pool.

## Ownership boundary

The native manager is the source of truth for all mutable cache metadata. Python keeps immutable request facts and compatibility handles for existing scheduler outputs, but it must not mirror reference counts, queue links, cache membership, or request block tables.

| Component | Owner | Responsibility |
| --- | --- | --- |
| Scheduler request state | Python | Token counts, request status, block hashes, scheduling policy, and model-facing output |
| Python KV adapter | Python | Configuration validation, conversion of request facts into native inputs, and wrapping returned block IDs |
| Block arena and free/LRU queue | Rust | Block identity, reference counts, cache metadata, eviction order, and free capacity |
| Hybrid coordinator | Rust | Cross-group hit reconciliation, atomic capacity planning, allocation, caching, skipped-block release, and request release |
| Full-attention policy | Rust | Contiguous prefix lookup, dense block-table growth, full-block caching, and common-prefix counting |
| Mamba `align` policy | Rust | Checkpoint lookup, sparse position-indexed block tables, rolling-state allocation, old-state release, and copy actions |
| GPU model runner | Python/CUDA | Consumption of block tables and execution of returned block-copy actions |

## Native data model

`BlockPool` stores a contiguous arena of block records and a shared intrusive free/LRU queue. Each block record contains its reference count, optional group-qualified hash metadata, free-queue links, and queue membership. Block zero remains the null block and is never placed in the free queue.

`CacheIndex` maps a key composed of the block hash and KV cache group ID to one or more block IDs. Keeping the group ID in the key preserves independent cache residency for identical tokens in different cache groups while retaining a single physical allocation pool and eviction queue.

`RequestState` stores one position-indexed block table per group, the number of cached blocks or checkpoint boundary per group, and Mamba `align` state such as the previous resident checkpoint position. Mamba positions without a resident state use the null block ID in the returned table, matching the current Python contract.

`PendingActions` stores block copies produced while advancing a Mamba `align` request. Actions are drained in a batch through the existing scheduler-facing copy API; Rust never invokes a Python callback while holding mutable manager state.

## Operation semantics

### Prefix lookup

The full-attention policy scans block hashes from the start and stops on the first miss. The Mamba policy searches for the latest reusable checkpoint no deeper than the current full-attention hit. The hybrid coordinator reconciles both results to a boundary valid for every group, truncates full-attention blocks to that boundary, and returns a sparse Mamba table ending at the matching checkpoint. Prefix hits remain aligned to the scheduler block size in the initial configuration.

### Allocation

The coordinator first computes the evictable hit blocks and new physical blocks required by both groups, applies reserved-block and watermark constraints once, and returns `None` without mutation if capacity is insufficient. On success it touches all local hit blocks before any eviction, creates the request state, grows the dense full-attention table, advances the sparse Mamba table, records required Mamba state copies, caches newly finalized blocks, and returns new block IDs grouped in model-facing order.

### Cache commit

Full-attention blocks are inserted into the group-qualified hash index when their token range is finalized. Mamba `align` checkpoints are cached only at boundaries permitted by the existing reachability rules. Duplicate hashes retain multiple physical block IDs until request release, matching the append-only block-table behavior of vLLM V1.

### Skipped-block release

When a Mamba `align` request advances, state blocks older than the copy source and current destination are released and their table positions become the null block. The source block remains referenced until its copy action is safe for the model runner to execute.

### Request release and eviction

Request release decrements group tables in reverse position order. Uncached blocks return to the free-queue head for immediate reuse, while cached blocks return to the tail to preserve LRU behavior. Allocating a cached free block removes its group-qualified hash entry before reuse.

### Common prefix

The full-attention group counts the leading blocks whose reference count equals the number of active request tables. The Mamba group returns zero because cascade attention does not consume Mamba checkpoints. Both values are produced from the same native request registry without reconstructing Python block objects.

## Python interface and FFI evolution

The compatibility adapter initially implements the existing `KVCacheManager` methods with high-level native calls so scheduler behavior can be compared without changing unrelated Python cache classes. Each native call completes an atomic manager transition and returns plain block IDs or batched actions; Python `KVCacheBlock` objects remain immutable ID handles only.

After hybrid correctness is established, the interface can add `plan_step`, accepting descriptors for all running and newly admitted requests in one scheduler iteration and returning lookup results, allocations, preemptions, per-group block-table updates, common-prefix counts, frees, and copy actions in one result. The native ownership model in this document is required from the first phase so adding `plan_step` changes the call shape rather than migrating state a second time.

## Supported initial configuration

- Exactly two KV cache groups: one `FullAttentionSpec` and one `MambaSpec` using `mamba_cache_mode="align"`.
- Prefix caching enabled, identical cache, scheduler, and hash block sizes, and DCP/PCP world size one.
- No connector, event publisher, metrics collector, zeroing, EAGLE/MTP, speculative Mamba blocks, deferred frees, or external computed blocks.
- A shared block pool with group-qualified cache keys and one null block.

The existing single-group Rust full-attention manager remains supported. Backend construction validates the complete configuration before allocating native state and names every unsupported feature in the error.

## Rust module layout

- `block_pool.rs` owns block records, the intrusive free/LRU queue, group-qualified cache indices, touching, eviction, and release.
- `full_attention.rs` owns full-attention lookup, dense table growth, caching, and common-prefix logic.
- `mamba.rs` owns `align` checkpoint lookup, sparse tables, rolling-state transitions, and copy actions.
- `hybrid.rs` owns request state, cross-group reconciliation, atomic capacity planning, operation ordering, and public native methods.
- `python.rs` exposes PyO3 classes and converts Python inputs and native outputs without owning cache state.

The implementation should share the block-pool and full-attention code between the unitary and hybrid managers instead of maintaining two independent allocators.

## Correctness invariants

- Every non-null block is in exactly one of two states: referenced by at least one request and absent from the free queue, or unreferenced and present exactly once in the free queue.
- Every cached block has one matching group-qualified cache-index entry, and eviction removes both directions of that relationship before the block is reused.
- A request table never references a block whose reference count is zero, except for the distinguished null block.
- A failed cross-group capacity check leaves block metadata, request tables, cache indices, pending actions, and queue order unchanged.
- Local hit blocks for every group are touched before allocating any block that may evict cache entries.
- A Mamba source state remains resident until the corresponding copy action has been emitted, and older positions are replaced with the null block before their physical blocks are reused.
- The reconciled hit length never exceeds any individual group hit and is aligned to the scheduler block size.

## Testing

Rust unit tests cover queue transitions, group-qualified duplicate hashes, cross-group capacity rollback, full and Mamba hit reconciliation, rolling Mamba state copies, reverse release order, eviction, reset, and common-prefix counting. Tests use deterministic fixtures and snapshot complete manager state where practical.

Python parity tests execute the same public manager scenarios against the Python and Rust backends and compare cached-token counts, per-group block tables, allocation failures, copy actions, usage, eviction, reset, and release behavior. Scheduler tests drive real `schedule()` and `update_from_output()` calls without a GPU.

The CPU performance test uses a 100K-token initial context and a 200-turn conversation with 100 input and 100 output tokens per turn. It reports the median across turns and the median of the final 25 turns for admission, decode schedule steps, finish updates, and total schedule time per turn. The performance gate compares matched medians and requires Rust to be strictly faster for every supported operation.

End-to-end validation uses the repository Rust `vllm-bench`, default asynchronous scheduling, and no `--enforce-eager`. Qwen3.5-0.8B is the first hybrid-Mamba target because its fast model execution is most likely to expose scheduler savings, followed by gpt-oss-20b after the sliding-window policy is implemented. Per-turn TTFT is compared at matching turn indices instead of averaging turns with different context lengths.

## Rollout

The backend remains opt-in behind `VLLM_USE_RUST_KV_CACHE_MANAGER`. Construction logs the selected native layout, and unsupported configurations fail closed. The Python implementation remains the reference behavior until parity, CPU scheduler performance, and model-level validation are complete. Sliding-window support and the batched `plan_step` interface are separate follow-up changes built on the same native block pool and hybrid coordinator.

## Risks

The highest correctness risk is cross-group partial mutation during allocation failure, so planning and mutation must be separate phases. The highest Mamba-specific risk is releasing a checkpoint before its copy action is consumed, so copy-source lifetime must be explicit in native state. The highest performance risk is preserving too many small Python-to-Rust calls, which is why the ownership boundary forbids Python-side metadata and the follow-up `plan_step` API is part of the design. The end-to-end risk is that asynchronous scheduling continues to hide CPU savings; this does not invalidate the manager speedup, but model validation must distinguish scheduler headroom from visible latency.
