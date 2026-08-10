# Fuzz Testing the V1 Core: Scheduler and KV Cache Manager

This document describes the hypothesis-based fuzz harness in
`tests/v1/fuzz/`: what it covers, how it works, and how to extend it.

## Goals

The v1 scheduler and KV cache manager already have strong deterministic
tests and a stdlib-random "blast" test
(`tests/v1/core/test_priority_scheduler_random.py`). The fuzz harness adds:

1. **Shrinkable reproductions.** Every input (operation sequence, request
   shapes, token contents, cache capacities) is drawn from hypothesis, so
   any failure is automatically minimized to a small replayable example.
2. **Oracles beyond self-consistency.** In addition to block-pool
   accounting invariants, the harness checks a committed-progress liveness
   oracle, an encoder-cache accounting oracle, and a same-trace
   differential between cache-on and cache-off KV cache managers.
3. **A wider state space.** Hybrid KV cache groups (full attention + GDN /
   Mamba), real async-scheduling overlap, prefix caching, speculative
   decoding, multimodal encoder-cache pressure, and preemption-heavy
   profiles, fuzzed in combination.

## Layout

- `tests/v1/fuzz/fuzz_utils.py` — shared factories, mock model outputs,
  and oracle checkers. Also registers the hypothesis profiles (see
  [Running](#running)).
- `tests/v1/fuzz/test_scheduler_statemachine.py` — a hypothesis
  `RuleBasedStateMachine` driving `Scheduler` / `AsyncScheduler` through
  random interleavings of add / step / abort / reset-prefix-cache.
- `tests/v1/fuzz/test_kv_cache_manager_fuzz.py` — a state machine driving
  `KVCacheManager` directly (no scheduler), with the cache-on/off
  differential.

## Scheduler state machine

Each test instance builds a scheduler from the configuration matrix:

- KV layout: single full-attention group, or hybrid full attention + GDN
  (`MambaSpec`) modeled after Qwen3.5-style models, with
  `mamba_cache_mode` aligned with the prefix-caching setting.
- `enable_prefix_caching` on/off; `num_speculative_tokens` in {None, 1, 5};
  standard vs. preemption-heavy profiles (small `num_blocks` to force
  preemption); sync vs. `AsyncScheduler`; default vs. tight encoder cache.

Rules draw their parameters from hypothesis (`st.data()`), including
request prompt lengths, priorities, multimodal layouts, token contents,
shared prefixes (to trigger prefix-cache hits), and encoder-cache
capacity. Multimodal identifiers are drawn from a small pool so distinct
requests genuinely share encoder-cache entries.

### Mock model output contract

The mock `ModelRunnerOutput` mirrors the real runner contract:

- Requests still mid chunked prefill produce an empty token list.
- Decode requests produce `1 + num_accepted` tokens, where
  `num_accepted` is bounded by the per-request
  `scheduled_spec_decode_tokens` from the `SchedulerOutput`.
- EOS / stop tokens only appear in the final position of a request's
  output, and stop tokens only when the request configured them.

Getting this contract right matters: a looser mock both hides real bugs
(e.g. never triggering speculative rejection) and creates false ones
(e.g. letting the scheduler derive negative rejection counts).

### Async scheduling

For `AsyncScheduler`, the state machine keeps a queue of in-flight
outputs: step N's output is processed after step N+1 has already been
scheduled, so `schedule()` always runs with one output in flight. This
exercises stale outputs, abort/preemption of in-flight requests, and
`num_in_flight_tokens` accounting. (The `defer_block_free` path requires
a KV consumer connector and is not covered yet.)

### Oracles

After every step, and at teardown:

- **SchedulerOutput validity**: new/cached request id sets,
  `num_scheduled_tokens` totals, finished-id subsets, freed mm hashes.
- **Block-pool accounting**: free queue has no duplicates and matches
  `get_num_free_blocks()`; every non-null block is either free
  (`ref_cnt == 0`) or held, never both; every cached hash maps to a valid
  block; for running requests, each held block's `ref_cnt` equals the
  exact number of running holders (works across hybrid KV groups since
  the pool is shared).
- **Encoder-cache accounting**: `num_freeable_slots == num_free_slots +
  sum(freeable)`; freeable keys are exactly the cache entries with an
  empty request-reference set; cache-side references are a subset of
  request-side records (the request side may be a strict superset: one
  request may reference the same identifier twice, which the per-request
  reference set cannot express). Teardown asserts
  `num_freeable_slots == cache_size` to catch reference leaks.
- **Committed-progress liveness**: a sliding window fails if there is
  pending work but zero committed progress — finished requests, delivered
  output tokens, or first-time prefill completions (computed minus
  in-flight tokens, so async only counts committed tokens; preemption
  rollbacks do not clear the mark). Scheduled-but-rolled-back tokens do
  not count as progress.
- **Teardown drain**: all requests are aborted and the block pool must
  return every block except the null block.

## KV cache manager differential

The KV manager state machine replays every drawn operation
(`get_computed_blocks` / `allocate_slots` / `free` / `cache_blocks` /
`reset_prefix_cache` / decode) against two independent managers, one with
prefix caching on and one off, and compares them per step:

- The cache-off side must always report zero computed blocks.
- The cache-on hit length is bounded above by ever-cached prefixes and
  below by still-live (pinned) cached prefixes. For hybrid configs the
  lower bound is checked on the full-attention group via
  `find_longest_cache_hit_per_group`, because the aggregate hit is a
  cross-group minimum and SWA groups legitimately cache less.
- Requests that never diverged (never decoded on exactly one side) must
  advance tokens identically on both sides.

Note that allocation success is intentionally *not* compared between
sides: pinned shared-prefix blocks can make the cache-on side succeed
where cache-off fails and vice versa.

## Running

```bash
# CI smoke profile (default): deterministic, 3 examples x 40 steps
.venv/bin/python -m pytest tests/v1/fuzz/ -v

# Deep profile: 30 examples x 400 steps, random seeds
HYPOTHESIS_PROFILE=long .venv/bin/python -m pytest tests/v1/fuzz/
```

Failing examples are persisted in the hypothesis database and replayed
first on the next run. All tests are marked `cpu_test`.

## Pitfalls discovered while building this

1. **Swarm-testing phase lock.** With multiple hypothesis `@rule`s and
   the deterministic profile, hypothesis's swarm-testing feature flags
   can disable a rule in every evaluation (the byte stream for an empty
   scheduler is nearly identical across rules). We measured
   `add_requests` disabled 47/47 times — the state machine was silently
   fuzzing an empty scheduler. Both state machines therefore use a
   single rule that draws the action (and its parameters) explicitly, so
   the action choice itself is shrinkable data. If you add rules, verify
   with a temporary `assert False` that they actually run.
2. **Mock contract drift** (see above) produces both false positives and
   false negatives; mirror `update_from_output`'s consumption logic.
3. **Oracles must match dedup semantics.** The encoder cache keys
   references per (request, hash); a request referencing one image twice
   is a legitimate state that a strict bidirectional equality rejects.

## Known gaps

- `defer_block_free` and KV connector paths (needs a consumer connector).
- Structured output / grammar bitmask scheduling.
- GPU-side components: the model runner and attention kernels are out of
  scope; the runner seam is the mocked `ModelRunnerOutput`.
- `use_v2_model_runner=True` changes `scheduled_new_reqs` semantics for
  resumed requests; the output-validity oracle does not handle that yet.
