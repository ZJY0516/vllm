# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's sequential Markov sampling, one launch per draft step.

`DSparkSpeculator._sample_sequential` runs, for each of the `k` draft steps,
the Markov embedding of the previous token, the bias GEMV over the whole
vocabulary (`markov_w2`: 163840 x 256 BF16, 84 MB, replicated), the add to the
base logits ([R, 163840] written twice), the Gumbel sample (its kernel, a torch
argmax over the block maxima and a gather) and the copy into `draft_tokens`:
about nine launches and three full-vocabulary [R, V] tensors per step.

Here each step is two launches. The step kernel's programs each own 256
vocabulary entries and 16 request rows: they gather the rows' Markov embeddings
(`markov_w1[prev]`), stream their `markov_w2` range in [64, rank] tiles and
multiply all rows at once on the tensor cores (FP32 accumulation), add the base
logits, write the step's logits into the draft logits cache and keep each row's
Gumbel-max over their range. A select launch (one program per row) takes the
first of the largest per-program maxima, which is the step's token: it writes
`draft_tokens` and the next step's previous token.

Exactness: the same sampling as vLLM's loop. The bias is the FP32 dot product of
the embedding and the `markov_w2` row rounded once to the logits dtype (the GEMV
output dtype), the add is one FP32 add rounded to the logits dtype (a BF16 tensor
add), the cache holds the logits before temperature, and the Gumbel noise is
vLLM's own draw (`murmur3` keyed by the request seed mixed with the draft step,
the position before the sampled token plus the draft salt, and the token id),
divided by the temperature exactly as `gumbel_noised_argmax` does. Only the
summation order of the 256-term bias dot product differs from cuBLAS's GEMV, as
any GEMM tiling may; ties keep the first token as vLLM's two-level argmax does.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _log1p_neg_stable,
    murmur3_uniform32,
    murmur3_uniform64,
    spec_step_seed,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row

logger = init_logger(__name__)

# Vocabulary rows of markov_w2 a tile multiplies at a time (tensor-core N).
VT = 64
_DRAFT_NOISE_SALT = 1 << 30


@triton.jit
def _add_rn(a, b):
    # One rounded FP32 add, kept apart from the dot: the compiler would otherwise
    # fold the base logits into the dot's accumulator (another summation order).
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gumbel_key(lf, temp, seed, pos, v, USE_FP64: tl.constexpr):
    """`gumbel_noised_argmax`'s key of FP32 logits `lf` (temperature applied
    in FP32, then the draft-salted noise; plain logits at temperature 0)."""
    noisy = temp != 0.0
    lf = tl.where(noisy, lf / temp, lf)
    if USE_FP64:
        lf = lf.to(tl.float64)
        u = murmur3_uniform64(seed, pos, v)
        g = -tl.log(-tl.log(u))
    else:
        u = murmur3_uniform32(seed, pos, v)
        g = -tl.log(-_log1p_neg_stable(u))
    return tl.where(noisy, lf + g, lf)


@triton.jit
def _gumbel_log_arg32(seed, pos, v):
    """The argument of the outer log of `_gumbel_key`'s FP32 noise
    `-log(-log1p(-u))`: the noise is `-log` of it. The step kernel takes that
    last log itself, so its add to the logits compiles as in
    `gumbel_noised_argmax` (a stored noise, added in another kernel, rounds
    differently: 1 ulp on about a fifth of the keys)."""
    u = murmur3_uniform32(seed, pos, v)
    return -_log1p_neg_stable(u)


@triton.jit
def _markov_noise_kernel(
    noise_ptr,  # [rows_cap, k, V] FP32
    noise_stride0,
    noise_stride1,
    idx_map_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    seeds_ptr,
    vocab,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    SALT: tl.constexpr,
):
    # Program (vocabulary block, row * K + step): the step's noise (up to its
    # last log) of one row, which depends only on the request seed, the position
    # and the token id.
    v = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    flat = tl.program_id(1)
    row = (flat // K).to(tl.int64)
    step = flat % K
    req = tl.load(idx_map_ptr + row * idx_stride + step).to(tl.int64)
    seed = spec_step_seed(tl.load(seeds_ptr + req, mask=req >= 0, other=0), step)
    pos = tl.load(pos_ptr + row * pos_stride + step) - 1
    pos = pos.to(tl.int64) + SALT
    arg = _gumbel_log_arg32(seed, pos, v)
    tl.store(
        noise_ptr + row * noise_stride0 + step * noise_stride1 + v, arg, mask=v < vocab
    )


@triton.jit
def _markov_step_kernel(
    prev_tok_ptr,  # [rows_cap] the previous step's tokens (select kernel)
    input_ids_ptr,
    anchor_idx_ptr,  # [rows] step 0's previous token: input_ids[anchor_idx[r]]
    w1_ptr,  # markov_w1 [V1, RANK]
    w2_ptr,  # markov_w2 [V, RANK]
    base_ptr,  # this step's base logits of row r at r * base_stride + v
    base_stride,
    cache_ptr,  # draft logits [rows, k, V]
    cache_stride0,
    cache_stride1,
    row_map_ptr,  # DraftLogitsCache.row_mapping or None
    idx_map_ptr,  # [rows, k] the row's request state index
    idx_stride,
    pos_ptr,  # [rows, k] the sampled token's position
    pos_stride,
    temp_ptr,
    seeds_ptr,
    out_max_ptr,  # [rows_cap, NB] this step's per-program maxima
    out_arg_ptr,
    noise_ptr,  # [rows_cap, k, V] FP32 ``_gumbel_log_arg32`` (``_markov_noise_kernel``)
    noise_stride0,
    noise_stride1,
    num_rows,
    vocab,
    NB,
    STEP: tl.constexpr,
    RANK: tl.constexpr,
    VB: tl.constexpr,  # vocabulary entries a program owns (multiple of VT)
    VT: tl.constexpr,
    RB: tl.constexpr,  # rows a program holds (tensor-core M)
    USE_FP64: tl.constexpr,
    SALT: tl.constexpr,
    NOISE: tl.constexpr,  # read the precomputed noise instead of hashing
):
    # Row chunks of one vocabulary range run as neighbouring programs, so the
    # range's markov_w2 tiles come from L2 for all but the first.
    pid = tl.program_id(1)
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    rmask = rows < num_rows
    rows64 = rows.to(tl.int64)
    # The previous token of each row: the anchor at step 0, else the select
    # launch's token of the previous step.
    if STEP == 0:
        anchor = tl.load(anchor_idx_ptr + rows, mask=rmask, other=0)
        prev = tl.load(input_ids_ptr + anchor, mask=rmask, other=0).to(tl.int64)
    else:
        prev = tl.load(prev_tok_ptr + rows64, mask=rmask, other=0)
    k = tl.arange(0, RANK)
    # [RB, RANK] Markov embeddings of the previous tokens.
    emb = tl.load(
        w1_ptr + prev[:, None] * RANK + k[None, :], mask=rmask[:, None], other=0.0
    )
    req = tl.load(idx_map_ptr + rows64 * idx_stride + STEP, mask=rmask, other=-1)
    req = req.to(tl.int64)
    valid = req >= 0
    crow = draft_logits_row(req, row_map_ptr)
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    seed = spec_step_seed(tl.load(seeds_ptr + req, mask=valid, other=0), STEP)
    pos = tl.load(pos_ptr + rows64 * pos_stride + STEP, mask=rmask, other=1) - 1
    pos = pos.to(tl.int64) + SALT
    cmask = rmask & valid & (crow >= 0)

    best_val = tl.full([RB], float("-inf"), tl.float64 if USE_FP64 else tl.float32)
    best_idx = tl.zeros([RB], tl.int64)
    v0 = pid * VB
    for t in range(0, VB, VT):
        v = v0 + t + tl.arange(0, VT)
        vmask = v < vocab
        w = tl.load(
            w2_ptr + v[:, None].to(tl.int64) * RANK + k[None, :],
            mask=vmask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        # The bias: one FP32 dot product per (row, token), rounded to the logits
        # dtype like the GEMV output; the add rounds once like a tensor add.
        acc = tl.dot(emb, tl.trans(w))
        bmask = rmask[:, None] & vmask[None, :]
        base = tl.load(
            base_ptr + rows64[:, None] * base_stride + v[None, :],
            mask=bmask,
            other=0.0,
            eviction_policy="evict_first",
        )
        dt = base.dtype
        logits = _add_rn(base.to(tl.float32), acc.to(dt).to(tl.float32)).to(dt)
        tl.store(
            cache_ptr
            + crow[:, None] * cache_stride0
            + STEP * cache_stride1
            + v[None, :],
            logits,
            mask=cmask[:, None] & vmask[None, :],
        )
        if NOISE:
            # ``_gumbel_key``'s FP32 path with the hash and the inner log read
            # back: the same operations, in the same kernel, from the outer log on.
            noisy = temp[:, None] != 0.0
            arg = tl.load(
                noise_ptr
                + rows64[:, None] * noise_stride0
                + STEP * noise_stride1
                + v[None, :],
                mask=bmask & noisy,
                other=1.0,
            )
            lf = logits.to(tl.float32)
            lf = tl.where(noisy, lf / temp[:, None], lf)
            x = tl.where(noisy, lf + -tl.log(arg), lf)
        else:
            x = _gumbel_key(
                logits.to(tl.float32),
                temp[:, None],
                seed[:, None],
                pos[:, None],
                v[None, :],
                USE_FP64,
            )
        x = tl.where(vmask[None, :], x, float("-inf"))
        bv, bi = tl.max(
            x, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        take = bv > best_val
        best_val = tl.where(take, bv, best_val)
        best_idx = tl.where(take, (v0 + t + bi).to(tl.int64), best_idx)
    tl.store(out_max_ptr + rows64 * NB + pid, best_val, mask=rmask)
    tl.store(out_arg_ptr + rows64 * NB + pid, best_idx, mask=rmask)


@triton.jit
def _markov_select_kernel(
    max_ptr,
    arg_ptr,
    prev_tok_ptr,
    tokens_ptr,
    tokens_stride,
    num_rows,
    NB,
    STEP: tl.constexpr,
    NB_POW2: tl.constexpr,
):
    # Row r's token of step STEP: the first of its largest per-program maxima.
    r = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, NB_POW2)
    m = tl.load(max_ptr + r * NB + j, mask=j < NB, other=float("-inf"))
    best = tl.argmax(m, axis=0, tie_break_left=True)
    tok = tl.load(arg_ptr + r * NB + best)
    tl.store(prev_tok_ptr + r, tok)
    tl.store(tokens_ptr + r * tokens_stride + STEP, tok)


# Vocabulary entries a program owns: 640 programs at 163840, about four per SM
# (GB300, one cold step, us: VB 256 -> 22, 1024 -> 40; a bare markov_w2 read
# streams at 15 us, bytes / 7.5 TB/s + the ~3.5 us kernel boundary).
VB = 256
# Request rows a program multiplies at once (the tensor-core M; more rows run as
# further programs over the same, then L2-resident, markov_w2 range).
RB = 16
# Sampled rows (requests x draft steps) up to which the Gumbel noise of all the
# steps is computed ahead, up to its last log, beside the lm_head
# (``_markov_noise_kernel``): the step kernels then read it instead of hashing
# [RB, VT] tiles whose rows are mostly padding (one request at c1: 16x the needed
# hashes and logs; GB300 ncu of a warm step: 41K cycles at 0.65 issued warps per
# scheduler, L2 at 17 %). They keep the last log, so that the keys round as in
# ``gumbel_noised_argmax`` (bit-equal; a fully precomputed noise is 1 ulp off on
# about a fifth of the keys). R1 k=4 graph with the noise kernel, us: 94 -> 82.
NOISE_ROWS = 32
NOISE_BLOCK = 1024


class FusedMarkov:
    """The fused Markov steps of one speculator (scratch sized for its largest
    batch, allocated once so captured graphs keep their addresses)."""

    def __init__(self, vocab, rank, max_rows, device, use_fp64):
        self.vocab, self.rank, self.use_fp64 = vocab, rank, use_fp64
        self.vb = VB
        self.nb = triton.cdiv(vocab, self.vb)
        self.nb_pow2 = triton.next_power_of_2(self.nb)
        dt = torch.float64 if use_fp64 else torch.float32
        self.bufs = (
            torch.empty(max_rows, self.nb, dtype=dt, device=device),
            torch.empty(max_rows, self.nb, dtype=torch.int64, device=device),
        )
        self.prev = torch.zeros(max_rows, dtype=torch.int64, device=device)
        self.max_rows = max_rows
        self.device = device
        self.noise = None  # [NOISE_ROWS * V] FP32, at the first noise launch
        self.stream = None

    def noise_rows(self, num_reqs, n_spec):
        """Whether this batch's Gumbel noise is precomputed (FP32 noise, at most
        `NOISE_ROWS` sampled rows)."""
        return not self.use_fp64 and 1 <= num_reqs * n_spec <= NOISE_ROWS

    def fork_noise(self, spec, num_reqs, n_spec):
        """Launch this draft step's Gumbel noise on a side stream (it depends only
        on the seeds, the positions and the token ids, not on any logits), so it
        runs beside the lm_head; `join` before the steps read it."""
        if self.noise is None:
            self.noise = torch.empty(
                NOISE_ROWS * self.vocab, dtype=torch.float32, device=self.device
            )
            self.stream = torch.cuda.Stream(device=self.device)
        main = torch.cuda.current_stream(self.device)
        assert self.stream is not None
        self.stream.wait_stream(main)
        noise = self.noise[: num_reqs * n_spec * self.vocab].view(
            num_reqs, n_spec, self.vocab
        )
        idx_map = spec.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[: num_reqs * n_spec].view(num_reqs, n_spec)
        with torch.cuda.stream(self.stream):
            grid = (triton.cdiv(self.vocab, NOISE_BLOCK), num_reqs * n_spec)
            _markov_noise_kernel[grid](
                noise,
                noise.stride(0),
                noise.stride(1),
                idx_map,
                idx_map.stride(0),
                sample_pos,
                sample_pos.stride(0),
                spec.seeds,
                self.vocab,
                K=n_spec,
                BLOCK=NOISE_BLOCK,
                SALT=_DRAFT_NOISE_SALT,
                num_warps=4,
            )
        return noise

    def join(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def run(self, spec, num_reqs, base_logits, w1, w2, noise=None):
        """`base_logits`: [num_reqs, k, V]; writes the draft logits cache and
        `spec.draft_tokens[:num_reqs]`. `noise`: `fork_noise`'s (joined), or
        None to hash the noise in the steps."""
        n_spec = base_logits.shape[1]
        use_noise = noise is not None
        if noise is None:
            noise = self.prev  # unused pointer (NOISE=False)
            noise_strides = (0, 0)
        else:
            noise_strides = (noise.stride(0), noise.stride(1))
        rb = RB
        grid = (triton.cdiv(num_reqs, rb), self.nb)
        idx_map = spec.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[: num_reqs * n_spec].view(num_reqs, n_spec)
        cache = spec.draft_logits.values
        tokens = spec.draft_tokens
        for step in range(n_spec):
            out_max, out_arg = self.bufs
            _markov_step_kernel[grid](
                self.prev,
                spec.input_buffers.input_ids,
                spec._anchor_idx,
                w1,
                w2,
                base_logits[:, step],
                base_logits.stride(0),
                cache,
                cache.stride(0),
                cache.stride(1),
                spec.draft_logits.row_mapping,
                idx_map,
                idx_map.stride(0),
                sample_pos,
                sample_pos.stride(0),
                spec.temperature,
                spec.seeds,
                out_max,
                out_arg,
                noise,
                *noise_strides,
                num_reqs,
                self.vocab,
                self.nb,
                STEP=step,
                RANK=self.rank,
                VB=self.vb,
                VT=VT,
                RB=rb,
                USE_FP64=self.use_fp64,
                SALT=_DRAFT_NOISE_SALT,
                NOISE=use_noise,
                num_warps=4,
                num_stages=2,
            )
            _markov_select_kernel[(num_reqs,)](
                out_max,
                out_arg,
                self.prev,
                tokens,
                tokens.stride(0),
                num_reqs,
                self.nb,
                STEP=step,
                NB_POW2=self.nb_pow2,
            )


def why_not(spec, model, num_reqs):
    """Why this draft step takes vLLM's loop, or None when the fused launches
    compute the same sampling."""
    head = model.model.markov_head
    if spec.draft_logits is None:
        return "greedy drafts"
    if spec.draft_watermarker is not None:
        return "a draft watermarker"
    if spec._d2t_scatter_index is not None:
        return "a reduced draft vocabulary"
    if spec.use_confidence_head:
        return "the confidence head"
    if spec.acceptance_estimator is not None:
        return "the acceptance estimator"
    if hasattr(model, "apply_output_logit_transform"):
        return "an output logit transform"
    lp = model.logits_processor
    if lp.scale != 1.0 or lp.soft_cap is not None:
        return "scaled logits"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "Markov weights not bf16"
    rank = w1.shape[1]
    if w2.shape[1] != rank or rank % 16 or rank & (rank - 1) or rank > 512:
        return f"Markov rank {rank}"
    if not (w1.is_contiguous() and w2.is_contiguous()):
        return "Markov weight layout"
    if head.markov_w2.quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    if num_reqs < 1:
        return "no requests"
    return None


_FUSED: dict = {}


def sample_sequential(spec, model, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """`DSparkSpeculator._sample_sequential` with the fused launches. Returns
    False, having done nothing, when this step takes vLLM's loop."""
    why = why_not(spec, model, num_reqs)
    if why is not None:
        logger.info_once("DSpark fused Markov sampling off: %s", why)
        return False
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    head = model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    sharded = _sharded(spec, model, w2, head_hidden.device)
    if sharded is not None and sharded.serves(spec, num_reqs, n_spec):
        sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
        local = sharded.local_logits(model, sample_hidden)
        if local.shape[-1] == sharded.vs:
            steps = sharded.steps
            if steps is not None and steps.serves(num_reqs, n_spec, local):
                steps.run(spec, num_reqs, local, w1, w2)
                return True
            noise = sharded.fork_noise(spec, num_reqs, n_spec)
            if noise is not None:
                sharded.join()
            sharded.run(spec, num_reqs, local, w1, w2, noise)
            return True
        sharded.disable(f"lm_head shard width {local.shape[-1]}")
    fused = _fused(spec, w1, w2, head_hidden.device)
    noise = None
    if fused.noise_rows(num_reqs, n_spec):
        noise = fused.fork_noise(spec, num_reqs, n_spec)
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base_logits = model.compute_draft_logits(sample_hidden)
    if noise is not None:
        fused.join()
    cache = spec.draft_logits.values
    vocab = base_logits.shape[-1]
    if (
        base_logits.dtype not in (torch.bfloat16, torch.float32)
        or base_logits.dtype != cache.dtype
        or base_logits.stride(-1) != 1
        or w2.shape[0] != vocab
        or cache.shape[-1] < vocab
    ):
        logger.info_once(
            "DSpark fused Markov sampling off: %s base logits, %s cache",
            base_logits.dtype,
            cache.dtype,
        )
        return _loop_rest(spec, model, num_reqs, base_logits)
    fused.run(spec, num_reqs, base_logits.view(num_reqs, n_spec, vocab), w1, w2, noise)
    return True


def _fused(spec, w1, w2, device):
    """The speculator's FusedMarkov (markov_w2's vocabulary; `sample_sequential`
    checks it against the base logits')."""
    vocab, rank = w2.shape[0], w1.shape[1]
    key = (device, vocab, rank, bool(spec.use_fp64_gumbel))
    fused = _FUSED.get(key)
    if fused is None or fused.max_rows < spec.max_num_reqs:
        fused = FusedMarkov(vocab, rank, spec.max_num_reqs, device, key[-1])
        _FUSED[key] = fused
        logger.info_once(
            "DSpark fused Markov sampling: vocabulary %d, rank %d, %d programs",
            vocab,
            rank,
            fused.nb,
        )
    return fused


def _loop_rest(spec, model, num_reqs, base_logits) -> bool:
    """The loop of vLLM from its base logits on (base logits computed once)."""
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    base_logits = base_logits.view(num_reqs, n_spec, -1)
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(n_spec):
        bias = model.markov_bias(model.markov_embed(prev))
        draft = spec._sample_logits(
            base_logits[:, i] + bias, idx_map[:, i], sample_pos[:, i], i
        )
        spec.draft_tokens[:num_reqs, i] = draft
        prev = draft
    return True


# --------------------------------------------------------------------------------
# Vocabulary-sharded steps (TP ranks)
# --------------------------------------------------------------------------------
@triton.jit
def _markov_noise_shard_kernel(
    noise_ptr,  # [rows_cap, k, V / TP] FP32: this rank's vocabulary range
    noise_stride0,
    noise_stride1,
    idx_map_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    seeds_ptr,
    vocab,  # the range's length
    voff,  # the range's first token id
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    SALT: tl.constexpr,
):
    # ``_markov_noise_kernel`` over [voff, voff + vocab): hashed with the global id.
    v = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    flat = tl.program_id(1)
    row = (flat // K).to(tl.int64)
    step = flat % K
    req = tl.load(idx_map_ptr + row * idx_stride + step).to(tl.int64)
    seed = spec_step_seed(tl.load(seeds_ptr + req, mask=req >= 0, other=0), step)
    pos = tl.load(pos_ptr + row * pos_stride + step) - 1
    pos = pos.to(tl.int64) + SALT
    arg = _gumbel_log_arg32(seed, pos, v + voff)
    tl.store(
        noise_ptr + row * noise_stride0 + step * noise_stride1 + v, arg, mask=v < vocab
    )


@triton.jit
def _markov_shard_step_kernel(
    prev_tok_ptr,
    input_ids_ptr,
    anchor_idx_ptr,
    w1_ptr,  # markov_w1 [V1, RANK]
    w2_ptr,  # this rank's markov_w2 rows [V / TP, RANK]
    base_ptr,  # this step's local base logits of row r at r * base_stride + v
    base_stride,
    out_ptr,  # local final logits [rows, k, V / TP]
    out_stride0,
    out_stride1,
    idx_map_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    temp_ptr,
    seeds_ptr,
    out_max_ptr,
    out_arg_ptr,
    noise_ptr,  # [rows_cap, k, V / TP] FP32 (``_markov_noise_shard_kernel``)
    noise_stride0,
    noise_stride1,
    num_rows,
    vocab,  # the range's length
    NB,
    voff,  # the range's first token id
    STEP: tl.constexpr,
    RANK: tl.constexpr,
    VB: tl.constexpr,
    VT: tl.constexpr,
    RB: tl.constexpr,
    SALT: tl.constexpr,
    NOISE: tl.constexpr,
):
    # ``_markov_step_kernel`` (FP32 keys) over the rank's range: the same bias, add,
    # rounding and key per token; the logits go to the local buffer (gathered into
    # the cache after the last step) and the maxima carry global token ids.
    pid = tl.program_id(1)
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    rmask = rows < num_rows
    rows64 = rows.to(tl.int64)
    if STEP == 0:
        anchor = tl.load(anchor_idx_ptr + rows, mask=rmask, other=0)
        prev = tl.load(input_ids_ptr + anchor, mask=rmask, other=0).to(tl.int64)
    else:
        prev = tl.load(prev_tok_ptr + rows64, mask=rmask, other=0)
    k = tl.arange(0, RANK)
    emb = tl.load(
        w1_ptr + prev[:, None] * RANK + k[None, :], mask=rmask[:, None], other=0.0
    )
    req = tl.load(idx_map_ptr + rows64 * idx_stride + STEP, mask=rmask, other=-1)
    req = req.to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    seed = spec_step_seed(tl.load(seeds_ptr + req, mask=valid, other=0), STEP)
    pos = tl.load(pos_ptr + rows64 * pos_stride + STEP, mask=rmask, other=1) - 1
    pos = pos.to(tl.int64) + SALT

    best_val = tl.full([RB], float("-inf"), tl.float32)
    best_idx = tl.zeros([RB], tl.int64)
    v0 = pid * VB
    for t in range(0, VB, VT):
        v = v0 + t + tl.arange(0, VT)
        vmask = v < vocab
        w = tl.load(
            w2_ptr + v[:, None].to(tl.int64) * RANK + k[None, :],
            mask=vmask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        acc = tl.dot(emb, tl.trans(w))
        bmask = rmask[:, None] & vmask[None, :]
        base = tl.load(
            base_ptr + rows64[:, None] * base_stride + v[None, :],
            mask=bmask,
            other=0.0,
            eviction_policy="evict_first",
        )
        dt = base.dtype
        logits = _add_rn(base.to(tl.float32), acc.to(dt).to(tl.float32)).to(dt)
        tl.store(
            out_ptr + rows64[:, None] * out_stride0 + STEP * out_stride1 + v[None, :],
            logits,
            mask=bmask,
        )
        if NOISE:
            noisy = temp[:, None] != 0.0
            arg = tl.load(
                noise_ptr
                + rows64[:, None] * noise_stride0
                + STEP * noise_stride1
                + v[None, :],
                mask=bmask & noisy,
                other=1.0,
            )
            lf = logits.to(tl.float32)
            lf = tl.where(noisy, lf / temp[:, None], lf)
            x = tl.where(noisy, lf + -tl.log(arg), lf)
        else:
            x = _gumbel_key(
                logits.to(tl.float32),
                temp[:, None],
                seed[:, None],
                pos[:, None],
                (v + voff)[None, :],
                False,
            )
        x = tl.where(vmask[None, :], x, float("-inf"))
        bv, bi = tl.max(
            x, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        take = bv > best_val
        best_val = tl.where(take, bv, best_val)
        best_idx = tl.where(take, (voff + v0 + t + bi).to(tl.int64), best_idx)
    tl.store(out_max_ptr + rows64 * NB + pid, best_val, mask=rmask)
    tl.store(out_arg_ptr + rows64 * NB + pid, best_idx, mask=rmask)


@triton.jit
def _markov_local_select_kernel(
    max_ptr, arg_ptr, packed_ptr, NB, NB_POW2: tl.constexpr
):
    # Row r's first largest per-program maximum on this rank -> (key bits, token).
    r = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, NB_POW2)
    m = tl.load(max_ptr + r * NB + j, mask=j < NB, other=float("-inf"))
    best = tl.argmax(m, axis=0, tie_break_left=True)
    key = tl.load(max_ptr + r * NB + best)
    tok = tl.load(arg_ptr + r * NB + best)
    tl.store(packed_ptr + r * 4, key.to(tl.int32, bitcast=True))
    tl.store(packed_ptr + r * 4 + 1, tok.to(tl.int32))


@triton.jit
def _markov_global_select_kernel(
    gathered_ptr,  # [rows, TP * 4] int32: every rank's (key bits, token, 0, 0)
    gathered_stride,
    prev_tok_ptr,
    tokens_ptr,
    tokens_stride,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
    STEP: tl.constexpr,
):
    # The first largest key over the ranks (rank order is token order): the step's
    # token, as the stock select's first of the largest per-program maxima.
    r = tl.program_id(0).to(tl.int64)
    w = tl.arange(0, W_POW2)
    bits = tl.load(gathered_ptr + r * gathered_stride + w * 4, mask=w < W, other=0)
    m = bits.to(tl.float32, bitcast=True)
    m = tl.where(w < W, m, float("-inf"))
    best = tl.argmax(m, axis=0, tie_break_left=True)
    tok = tl.load(gathered_ptr + r * gathered_stride + best * 4 + 1).to(tl.int64)
    tl.store(prev_tok_ptr + r, tok)
    tl.store(tokens_ptr + r * tokens_stride + STEP, tok)


@triton.jit
def _markov_cache_scatter_kernel(
    src_ptr,  # [rows * k, V] gathered final logits
    src_stride,
    cache_ptr,
    cache_stride0,
    cache_stride1,
    row_map_ptr,
    idx_map_ptr,
    idx_stride,
    vocab,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The stock step kernel's cache store, from the gathered logits: row (r, step)
    # into the request's cache row, nothing for a row without one.
    flat = tl.program_id(0)
    r = (flat // K).to(tl.int64)
    step = flat % K
    req = tl.load(idx_map_ptr + r * idx_stride + step).to(tl.int64)
    crow = draft_logits_row(req, row_map_ptr)
    ok = (req >= 0) & (crow >= 0)
    v = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = (v < vocab) & ok
    x = tl.load(src_ptr + flat.to(tl.int64) * src_stride + v, mask=m)
    tl.store(cache_ptr + crow * cache_stride0 + step * cache_stride1 + v, x, mask=m)


SCATTER_BLOCK = 2048
SHARD_VB = 64


class ShardedMarkov:
    """The vocabulary-sharded fused Markov steps of one speculator over the
    drafter's TP mailboxes (`comm.DrafterComm`). Built eagerly (collective: two
    small Lamport mailboxes) at the first draft outside a graph capture."""

    def __init__(self, comm, vocab, max_rows, device):
        from .comm import PAIR_SENT, ROWS_CAP, LamportRows, compile_gather

        self.comm, self.tp, self.rank = comm, comm.tp, comm.rank
        self.vocab, self.vs = vocab, vocab // comm.tp
        self.voff = self.rank * self.vs
        # More, smaller programs over the 1 / TP range (64 entries each: the launch
        # sweep's fastest; the per-entry math and the first-max order do not change).
        self.vb = SHARD_VB
        self.nb = triton.cdiv(self.vs, self.vb)
        self.nb_pow2 = triton.next_power_of_2(self.nb)
        # Requests per call (the per-request scratch) and sampled rows (R * k: the
        # logits gather's mailbox rows).
        self.max_rows = min(max_rows, ROWS_CAP)
        self.rows_cap = ROWS_CAP
        self.calls = 0
        self.off = False
        self.device = device
        rows = self.max_rows
        self.out_max = torch.empty(rows, self.nb, dtype=torch.float32, device=device)
        self.out_arg = torch.empty(rows, self.nb, dtype=torch.int64, device=device)
        self.prev = torch.zeros(rows, dtype=torch.int64, device=device)
        self.packed = torch.zeros(rows, 4, dtype=torch.int32, device=device)
        self.gathered = torch.empty(rows, 4 * self.tp, dtype=torch.int32, device=device)
        self.boxes = [
            LamportRows(comm.group, device, (ROWS_CAP, 4 * self.tp), PAIR_SENT)
            for _ in range(2)
        ]
        self.exchange = compile_gather(
            self.tp, 4, ROWS_CAP, PAIR_SENT, True, True, device
        )
        self.noise = None
        self.stream = None
        self.steps = None  # ShardedMarkovSteps (K31_MK1), built by `_sharded`
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=comm.group)

    def serves(self, spec, num_reqs, n_spec):
        if self.off:
            return False
        rows = num_reqs * n_spec
        cache = spec.draft_logits.values
        return (
            1 <= num_reqs <= self.max_rows
            and rows <= self.rows_cap
            and cache.dtype == torch.bfloat16
            and cache.shape[-1] >= self.vocab
        )

    def fork_noise(self, spec, num_reqs, n_spec):
        """This rank's range of the step noise on a side stream (`fork_noise`),
        or None when the steps hash it (more than `NOISE_ROWS` sampled rows)."""
        if not 1 <= num_reqs * n_spec <= NOISE_ROWS:
            return None
        if self.noise is None:
            self.noise = torch.empty(
                NOISE_ROWS * self.vs, dtype=torch.float32, device=self.device
            )
            self.stream = torch.cuda.Stream(device=self.device)
        assert self.stream is not None
        main = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(main)
        noise = self.noise[: num_reqs * n_spec * self.vs].view(
            num_reqs, n_spec, self.vs
        )
        idx_map = spec.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[: num_reqs * n_spec].view(num_reqs, n_spec)
        with torch.cuda.stream(self.stream):
            grid = (triton.cdiv(self.vs, NOISE_BLOCK), num_reqs * n_spec)
            _markov_noise_shard_kernel[grid](
                noise,
                noise.stride(0),
                noise.stride(1),
                idx_map,
                idx_map.stride(0),
                sample_pos,
                sample_pos.stride(0),
                spec.seeds,
                self.vs,
                self.voff,
                K=n_spec,
                BLOCK=NOISE_BLOCK,
                SALT=_DRAFT_NOISE_SALT,
                num_warps=4,
            )
        return noise

    def join(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def local_logits(self, model, hidden):
        """The lm_head shard's base logits [M, V / TP] (`draft_logits` without its
        vocabulary gather)."""
        cached = getattr(model, "_draft_lm_head_gemm", None)
        gemm = cached[1] if cached is not None else None
        x = hidden.contiguous()
        if gemm is not None:
            out = torch.empty(x.shape[0], gemm.n, dtype=x.dtype, device=x.device)
            gemm.run(x, out)
            return out
        return model.logits_processor._apply_head(model.lm_head, x, None)

    def disable(self, reason):
        logger.info_once("DSpark Markov sampling stays replicated: %s", reason)
        self.off = True

    def run(self, spec, num_reqs, local, w1, w2, noise):
        n_spec = local.shape[0] // num_reqs
        vs = self.vs
        self.calls += 1
        base = local.view(num_reqs, n_spec, vs)
        final = torch.empty_like(local)
        out = final.view(num_reqs, n_spec, vs)
        if noise is None:
            noise_t, ns = self.prev, (0, 0)
        else:
            noise_t, ns = noise, (noise.stride(0), noise.stride(1))
        w2s = w2[self.voff : self.voff + vs]
        idx_map = spec.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[: num_reqs * n_spec].view(num_reqs, n_spec)
        tokens = spec.draft_tokens
        grid = (triton.cdiv(num_reqs, RB), self.nb)
        packed = self.packed[:num_reqs]
        gathered = self.gathered[:num_reqs]
        for step in range(n_spec):
            _markov_shard_step_kernel[grid](
                self.prev,
                spec.input_buffers.input_ids,
                spec._anchor_idx,
                w1,
                w2s,
                base[:, step],
                base.stride(0),
                out,
                out.stride(0),
                out.stride(1),
                idx_map,
                idx_map.stride(0),
                sample_pos,
                sample_pos.stride(0),
                spec.temperature,
                spec.seeds,
                self.out_max,
                self.out_arg,
                noise_t,
                *ns,
                num_reqs,
                vs,
                self.nb,
                self.voff,
                STEP=step,
                RANK=w2.shape[1],
                VB=self.vb,
                VT=VT,
                RB=RB,
                SALT=_DRAFT_NOISE_SALT,
                NOISE=noise is not None,
                num_warps=4,
                num_stages=2,
            )
            _markov_local_select_kernel[(num_reqs,)](
                self.out_max, self.out_arg, packed, self.nb, NB_POW2=self.nb_pow2
            )
            box, mc = self.boxes[step % 2].view()
            self.exchange(packed, gathered, box, mc, self.rank, 1)
            _markov_global_select_kernel[(num_reqs,)](
                gathered,
                gathered.stride(0),
                self.prev,
                tokens,
                tokens.stride(0),
                W=self.tp,
                W_POW2=triton.next_power_of_2(self.tp),
                STEP=step,
            )
        full = self.comm.gather_logits(final)
        cache = spec.draft_logits.values
        _markov_cache_scatter_kernel[
            (num_reqs * n_spec, triton.cdiv(self.vocab, SCATTER_BLOCK))
        ](
            full,
            full.stride(0),
            cache,
            cache.stride(0),
            cache.stride(1),
            spec.draft_logits.row_mapping,
            idx_map,
            idx_map.stride(0),
            self.vocab,
            K=n_spec,
            BLOCK=SCATTER_BLOCK,
            num_warps=4,
        )


_SHARDED: dict = {}


def _sharded(spec, model, w2, device):
    """The speculator's ShardedMarkov, or None when the sampling stays replicated:
    no drafter mailboxes (TP 1, no multicast), a head other than the plain BF16
    vocab-parallel one, a padded or uneven vocabulary, FP64 Gumbel keys, or a first
    use under a graph capture (the mailboxes are built eagerly)."""
    import os

    if os.environ.get("VLLM_K3_SHARDED_MARKOV", "1") == "0":
        return None
    comm = getattr(model, "_drafter_comm", None)
    if comm is None:
        logger.info_once(
            "DSpark Markov sampling stays replicated: no drafter mailboxes"
        )
        return None
    vocab = w2.shape[0]
    lp, head = model.logits_processor, model.lm_head
    checks = (
        ("no drafter mailboxes", comm is None),
        ("no logits mailbox", comm is not None and comm.logits is None),
        ("TP 1", comm is not None and comm.tp <= 1),
        ("FP64 Gumbel keys", bool(spec.use_fp64_gumbel)),
        ("uneven vocabulary", comm is not None and vocab % comm.tp != 0),
        ("vocab shard", comm is not None and vocab // comm.tp != comm.vocab_shard),
        ("padded vocabulary", lp.org_vocab_size != vocab),
        ("lm_head TP", comm is not None and getattr(head, "tp_size", 1) != comm.tp),
        ("logits as input", bool(lp.logits_as_input)),
        ("head dtype", getattr(lp, "head_dtype", None) not in (None, torch.bfloat16)),
        ("head weight dtype", head.weight.dtype != torch.bfloat16),
    )
    for reason, failed in checks:
        if failed:
            logger.info_once("DSpark Markov sampling stays replicated: %s", reason)
            return None
    cached = getattr(model, "_draft_lm_head_gemm", None)
    if cached is None or cached[0] is not head.weight:
        # draft_logits' choice: DraftGemm for a plain BF16 head, else the head's GEMM.
        from .linears import DraftGemm

        plain = (
            type(head.quant_method).__name__ == "UnquantizedEmbeddingMethod"
            and head.weight.dim() == 2
        )
        model._draft_lm_head_gemm = (
            head.weight,
            DraftGemm(head.weight) if plain else None,
        )
    key = (device, vocab, comm.tp, comm.rank)
    sharded = _SHARDED.get(key)
    if sharded is None:
        if torch.cuda.is_current_stream_capturing():
            logger.info_once(
                "DSpark Markov sampling stays replicated: first use in a capture"
            )
            return None
        sharded = ShardedMarkov(comm, vocab, spec.max_num_reqs, device)
        _SHARDED[key] = sharded
        logger.info_once(
            "DSpark Markov sampling sharded over %d ranks (%d tokens each)",
            comm.tp,
            sharded.vs,
        )
        if os.environ.get(STEPS_ENV, "0") == "1":
            # Collective: every rank builds it here, at the same first draft.
            head_w1 = model.model.markov_head.markov_w1.weight
            sharded.steps = ShardedMarkovSteps.build(
                sharded, spec, spec.num_speculative_steps, head_w1, w2
            )
    return sharded


# --------------------------------------------------------------------------------
# Vocabulary-sharded steps, all draft steps in one launch (K31_MK1=1)
# --------------------------------------------------------------------------------
STEPS_ENV = "K31_MK1"
STEPS_CTAS_ENV = "K31_MK1_CTAS"  # CTA count override (tuning), 0 = default
# Variant (tests): default one tile per CTA only; "any" also fewer CTAs (each over
# several tiles).
STEPS_MODE_ENV = "K31_MK1_MODE"
STEPS_VT_ENV = "K31_MK1_VT"  # vocabulary entries per tile (tuning), default VT
STEPS_WARPS_ENV = "K31_MK1_WARPS"  # warps per CTA (tuning), default 4
# Requests the one-launch steps serve (more take ShardedMarkov's launches).
STEPS_RMAX = 4
# A rank's best (key, token) of a row as one ordered word: the key's order bits
# (32) above the inverted token id (18): the integer maximum is the largest key and,
# among equal keys, the smallest token, as the stock first-max argmax. A mailbox
# word carries the call's tag above them (bits 50-62, so the word stays positive).
_TOKEN_BITS = tl.constexpr(18)
_TAG_SHIFT = tl.constexpr(50)
_TAG_MOD = tl.constexpr(8191)


@triton.jit
def _store_sys_b64(addr, value, ok):
    # A relaxed system-scope 64-bit store to `addr` where `ok` (peer mailboxes).
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.s32 p, $3, 0; "
        "@p st.relaxed.sys.global.b64 [$1], $2; mov.b64 $0, $2; }",
        "=l,l,l,r",
        [addr, value, ok],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _launch_dependents():
    # griddepcontrol.launch_dependents (a no-op for a dependent without PDL).
    tl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // $0",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


# Every integer but the request count and the rank is fixed for a speculator, so
# the launched kernel is the one compiled (and checked) at build: specialized on
# them (measured faster in A/B: graph replay of the 16-rank K=7 call).
@triton.jit(do_not_specialize=["num_rows", "rank"])
def _markov_shard_steps_kernel(
    input_ids_ptr,
    anchor_idx_ptr,
    w1_ptr,  # markov_w1 [V1, RANK]
    w2_ptr,  # this rank's markov_w2 rows [V / TP, RANK]
    base_ptr,  # local base logits of (row, step) at row * s0 + step * s1 + v
    base_stride0,
    base_stride1,
    out_ptr,  # local final logits [rows, k, V / TP]
    out_stride0,
    out_stride1,
    idx_map_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    temp_ptr,
    seeds_ptr,
    noise_ptr,  # [RMAX, K, V / TP] FP32 scratch
    words_ptr,  # [2, K, RMAX, ctas] int64: every program's tagged best word
    count_ptr,  # [K] int64 arrivals per step (only grows)
    box_ptr,  # this rank's mailbox [2, K, RMAX, W] int64
    peers_ptr,  # [W] int64: every rank's mailbox address
    tokens_ptr,
    tokens_stride,
    num_rows,
    vocab,  # the range's length
    voff,  # the range's first token id
    num_tiles,
    rank,
    ctas,
    K: tl.constexpr,
    W: tl.constexpr,
    W_POW2: tl.constexpr,
    RANK: tl.constexpr,
    VT: tl.constexpr,
    RB: tl.constexpr,
    RMAX: tl.constexpr,
    SALT: tl.constexpr,
    CTAS_POW2: tl.constexpr,
    RK_POW2: tl.constexpr,  # a power of two >= RMAX * K
    HOIST: tl.constexpr,  # one tile per program, its markov_w2 tile loaded once
):
    # The K steps of `ShardedMarkov.run` in one grid of `ctas` programs, each owning
    # the VT-entry tiles pid, pid + ctas, ... of the rank's range. Per step and tile:
    # ``_markov_shard_step_kernel``'s bias, add, rounding, cache-bound logits and key
    # (FP32 keys, the noise up to its last log read back as with
    # ``_markov_noise_shard_kernel``); the program's first-max over its tiles as an
    # ordered word (key order bits, inverted token), tagged with the call. The step's
    # token is the largest word over the ranks (`_markov_global_select_kernel`'s
    # first largest key): the last program of the rank to arrive (step counter)
    # takes the largest of the rank's words and stores it into this rank's slot of
    # every rank's mailbox; every program polls the W slots. The waits (the words
    # of this rank's programs, the mailbox) are on programs of the same grid on
    # every rank, all resident (cooperative launch, checked when built): other
    # kernels can delay them but not block them.
    pid = tl.program_id(0)
    # The call: every program adds 1 to the step-0 counter per call and this one
    # has not yet, so it reads [call * ctas, call * ctas + ctas - 1].
    call = tl.load(count_ptr, volatile=True) // ctas
    tag = call % _TAG_MOD + 1
    par = call % 2
    rows = tl.arange(0, RB)
    rmask = rows < num_rows
    rows64 = rows.to(tl.int64)
    k = tl.arange(0, RANK)
    r4 = tl.arange(0, RMAX)
    j = tl.arange(0, W_POW2)
    g = tl.arange(0, CTAS_POW2)
    slot_mask = (r4[:, None] < num_rows) & (j[None, :] < W)
    peers = tl.load(peers_ptr + j, mask=j < W, other=0)
    # Step 0's previous token, and (one tile per program) the program's markov_w2
    # tile and step 0's inputs that do not depend on a token: issued first, they
    # load while the noise is checked.
    anchor = tl.load(anchor_idx_ptr + rows, mask=rmask, other=0)
    prev = tl.load(input_ids_ptr + anchor, mask=rmask, other=0).to(tl.int64)
    if HOIST:
        vh = pid * VT + tl.arange(0, VT)
        vhmask = vh < vocab
        hmask = rmask[:, None] & vhmask[None, :]
        wh = tl.load(
            w2_ptr + vh[:, None].to(tl.int64) * RANK + k[None, :],
            mask=vhmask[:, None],
            other=0.0,
        )
        req_n = tl.load(idx_map_ptr + rows64 * idx_stride, mask=rmask, other=-1)
        req_n = req_n.to(tl.int64)
        temp_n = tl.load(temp_ptr + req_n, mask=req_n >= 0, other=0.0).to(tl.float32)
        base_n = tl.load(
            base_ptr + rows64[:, None] * base_stride0 + vh[None, :],
            mask=hmask,
            other=0.0,
            eviction_policy="evict_first",
        )
    # The noise of the sampled rows (zero temperature: no noise), every step.
    rk = tl.arange(0, RK_POW2)
    rkm = rk < num_rows * K
    rk_req = tl.load(
        idx_map_ptr + (rk // K).to(tl.int64) * idx_stride + rk % K, mask=rkm, other=-1
    ).to(tl.int64)
    rk_temp = tl.load(temp_ptr + rk_req, mask=rkm & (rk_req >= 0), other=0.0)
    if tl.sum(tl.where(rk_temp.to(tl.float32) != 0.0, 1, 0)) > 0:
        for rs in range(0, num_rows * K):
            r = rs // K
            st = rs % K
            req = tl.load(idx_map_ptr + r.to(tl.int64) * idx_stride + st).to(tl.int64)
            t_rs = tl.load(temp_ptr + req, mask=req >= 0, other=0.0).to(tl.float32)
            if t_rs != 0.0:
                seed = spec_step_seed(
                    tl.load(seeds_ptr + req, mask=req >= 0, other=0), st
                )
                pos = tl.load(pos_ptr + r.to(tl.int64) * pos_stride + st) - 1
                pos = pos.to(tl.int64) + SALT
                row = noise_ptr + (r * K + st).to(tl.int64) * vocab
                for t in range(pid, num_tiles, ctas):
                    v = t * VT + tl.arange(0, VT)
                    arg = _gumbel_log_arg32(seed, pos, v + voff)
                    tl.store(row + v, arg, mask=v < vocab)
        tl.debug_barrier()
    if HOIST:
        arg_n = tl.load(
            noise_ptr + (rows64[:, None] * K) * vocab + vh[None, :],
            mask=hmask & (temp_n[:, None] != 0.0),
            other=1.0,
        )
    for step in range(0, K):
        emb = tl.load(
            w1_ptr + prev[:, None] * RANK + k[None, :], mask=rmask[:, None], other=0.0
        )
        if HOIST:
            temp = temp_n
        else:
            req = tl.load(
                idx_map_ptr + rows64 * idx_stride + step, mask=rmask, other=-1
            )
            req = req.to(tl.int64)
            temp = tl.load(temp_ptr + req, mask=req >= 0, other=0.0).to(tl.float32)
        noisy = temp[:, None] != 0.0
        best_val = tl.full([RB], float("-inf"), tl.float32)
        best_idx = tl.zeros([RB], tl.int64)
        if HOIST:
            acc = tl.dot(emb, tl.trans(wh))
            base = base_n
            dt = base.dtype
            hlogits = _add_rn(base.to(tl.float32), acc.to(dt).to(tl.float32)).to(dt)
            arg = arg_n
            lf = hlogits.to(tl.float32)
            lf = tl.where(noisy, lf / temp[:, None], lf)
            x = tl.where(noisy, lf + -tl.log(arg), lf)
            x = tl.where(vhmask[None, :], x, float("-inf"))
            bv, bi = tl.max(
                x, axis=1, return_indices=True, return_indices_tie_break_left=True
            )
            take = bv > best_val
            best_val = tl.where(take, bv, best_val)
            best_idx = tl.where(take, (voff + pid * VT + bi).to(tl.int64), best_idx)
        else:
            for t in range(pid, num_tiles, ctas):
                v = t * VT + tl.arange(0, VT)
                vmask = v < vocab
                w = tl.load(
                    w2_ptr + v[:, None].to(tl.int64) * RANK + k[None, :],
                    mask=vmask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc = tl.dot(emb, tl.trans(w))
                bmask = rmask[:, None] & vmask[None, :]
                base = tl.load(
                    base_ptr
                    + rows64[:, None] * base_stride0
                    + step * base_stride1
                    + v[None, :],
                    mask=bmask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                dt = base.dtype
                logits = _add_rn(base.to(tl.float32), acc.to(dt).to(tl.float32)).to(dt)
                tl.store(
                    out_ptr
                    + rows64[:, None] * out_stride0
                    + step * out_stride1
                    + v[None, :],
                    logits,
                    mask=bmask,
                )
                arg = tl.load(
                    noise_ptr + (rows64[:, None] * K + step) * vocab + v[None, :],
                    mask=bmask & noisy,
                    other=1.0,
                )
                lf = logits.to(tl.float32)
                lf = tl.where(noisy, lf / temp[:, None], lf)
                x = tl.where(noisy, lf + -tl.log(arg), lf)
                x = tl.where(vmask[None, :], x, float("-inf"))
                bv, bi = tl.max(
                    x, axis=1, return_indices=True, return_indices_tie_break_left=True
                )
                take = bv > best_val
                best_val = tl.where(take, bv, best_val)
                best_idx = tl.where(take, (voff + t * VT + bi).to(tl.int64), best_idx)
        # The ordered word: -0 as +0 (equal keys), then the float order as unsigned.
        bits = best_val.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        bits = tl.where(bits == 0x80000000, 0, bits)
        order = tl.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)
        packed = (order << _TOKEN_BITS) | ((1 << _TOKEN_BITS) - 1 - best_idx)
        slot = (par * K + step) * RMAX
        for i in tl.static_range(RMAX):
            if i < num_rows:
                word = tl.max(tl.where(rows == i, packed, 0), axis=0)
                # A strong store (the last program polls it), its old value unused.
                tl.atomic_xchg(
                    words_ptr + (slot + i) * ctas + pid,
                    (tag << _TAG_SHIFT) | word,
                    sem="relaxed",
                    scope="gpu",
                )
        old = tl.atomic_add(count_ptr + step, 1, sem="relaxed", scope="gpu")
        if old % ctas == ctas - 1:
            # The last program of this rank: every word of the rank (tagged, so
            # stores still on their way are waited for), their largest to every rank.
            best = tl.zeros([RMAX], tl.int64)
            for i in tl.static_range(RMAX):
                if i < num_rows:
                    ws = words_ptr + (slot + i) * ctas + g
                    gm = g < ctas
                    got = tl.load(ws, mask=gm, other=0, volatile=True)
                    miss = tl.sum(tl.where(gm & ((got >> _TAG_SHIFT) != tag), 1, 0))
                    while miss > 0:
                        got = tl.load(ws, mask=gm, other=0, volatile=True)
                        miss = tl.sum(tl.where(gm & ((got >> _TAG_SHIFT) != tag), 1, 0))
                    # Read again: every word is final now, in every thread's copy.
                    got = tl.load(ws, mask=gm, other=0, volatile=True)
                    best = tl.where(r4 == i, tl.max(tl.where(gm, got, 0), axis=0), best)
            addr = peers[None, :] + ((slot + r4[:, None]) * W + rank) * 8
            data = best[:, None] + tl.zeros([RMAX, W_POW2], tl.int64)
            _store_sys_b64(addr, data, slot_mask.to(tl.int32))
        if HOIST:
            nxt = step + 1
            more = nxt < K
            req_n = tl.load(
                idx_map_ptr + rows64 * idx_stride + nxt, mask=rmask & more, other=-1
            )
            req_n = req_n.to(tl.int64)
            temp_n = tl.load(temp_ptr + req_n, mask=req_n >= 0, other=0.0)
            temp_n = temp_n.to(tl.float32)
            base_n = tl.load(
                base_ptr
                + rows64[:, None] * base_stride0
                + nxt * base_stride1
                + vh[None, :],
                mask=hmask & more,
                other=0.0,
                eviction_policy="evict_first",
            )
            arg_n = tl.load(
                noise_ptr + (rows64[:, None] * K + nxt) * vocab + vh[None, :],
                mask=hmask & more & (temp_n[:, None] != 0.0),
                other=1.0,
            )
        xs = box_ptr + (slot + r4[:, None]) * W + j[None, :]
        got = tl.load(xs, mask=slot_mask, other=0, volatile=True)
        miss = tl.sum(tl.where(slot_mask & ((got >> _TAG_SHIFT) != tag), 1, 0))
        while miss > 0:
            got = tl.load(xs, mask=slot_mask, other=0, volatile=True)
            miss = tl.sum(tl.where(slot_mask & ((got >> _TAG_SHIFT) != tag), 1, 0))
        # Read again: every slot is final now, in every thread's copy of it.
        got = tl.load(xs, mask=slot_mask, other=0, volatile=True)
        top = tl.max(tl.where(slot_mask, got & ((1 << _TAG_SHIFT) - 1), -1), axis=1)
        tok = (1 << _TOKEN_BITS) - 1 - (top & ((1 << _TOKEN_BITS) - 1))
        if HOIST:
            # The cache-bound logits, off the token's path (read after the grid).
            tl.store(
                out_ptr
                + rows64[:, None] * out_stride0
                + step * out_stride1
                + vh[None, :],
                hlogits,
                mask=hmask,
            )
        tl.store(
            tokens_ptr + r4 * tokens_stride + step,
            tok,
            mask=(r4 < num_rows) & (pid == 0),
        )
        prev = tl.sum(tl.where(rows[:, None] == r4[None, :], tok[None, :], 0), axis=1)
    # The logits gather that follows (launched with PDL) may start: every program
    # of this grid is resident, and it waits for this grid's end to read.
    _launch_dependents()


def _resident_ctas(kernel, device) -> int:
    """CTAs of the compiled `kernel` the device holds at once (the occupancy API
    times the SM count), 0 when unknown."""
    import ctypes

    try:
        if getattr(kernel, "function", None) is None:
            kernel._init_handles()
        lib = ctypes.CDLL("libcuda.so.1")
        blocks = ctypes.c_int(0)
        rc = lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks),
            ctypes.c_void_p(int(kernel.function)),
            ctypes.c_int(32 * int(kernel.metadata.num_warps)),
            ctypes.c_size_t(int(kernel.metadata.shared)),
        )
        if rc != 0:
            logger.warning("DSpark Markov one-launch steps: occupancy rc %d", rc)
            return 0
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        return int(blocks.value) * int(sms)
    except Exception as exc:  # no occupancy answer: the launches stay as they are
        logger.warning("DSpark Markov one-launch steps: occupancy %s", exc)
        return 0


class ShardedMarkovSteps:
    """`ShardedMarkov.run` with all draft steps in one cooperative launch
    (`_markov_shard_steps_kernel`), then the same logits gather and cache scatter.
    Built eagerly (collective) next to its ShardedMarkov; `ok` is the same on every
    rank (MIN over the TP group), so every rank takes the same path for a step."""

    def __init__(self, sharded, n_spec):
        import os

        from ..symm import peer_rendezvous

        comm = sharded.comm
        self.sharded, self.comm, self.k = sharded, comm, n_spec
        self.tp, self.rank, self.device = comm.tp, comm.rank, sharded.device
        self.vs, self.voff, self.vocab = sharded.vs, sharded.voff, sharded.vocab
        self.vt = int(os.environ.get(STEPS_VT_ENV, "0") or 0) or VT
        self.tiles = triton.cdiv(self.vs, self.vt)
        self.ctas = 1
        dev = self.device
        self.noise = torch.zeros(
            STEPS_RMAX * n_spec * self.vs, dtype=torch.float32, device=dev
        )
        self.words = torch.zeros(
            2 * n_spec * STEPS_RMAX * self.tiles, dtype=torch.int64, device=dev
        )
        self.count = torch.zeros(n_spec, dtype=torch.int64, device=dev)
        import torch.distributed._symmetric_memory as symm_mem

        self.box = symm_mem.empty(
            2 * n_spec * STEPS_RMAX * self.tp, dtype=torch.int64, device=dev
        )
        self.box.zero_()
        handle = peer_rendezvous(self.box, comm.group)
        self.peers = torch.tensor(
            [int(p) for p in handle.buffer_ptrs], dtype=torch.int64, device=dev
        )
        self._handle = handle
        self.kernel = None
        self.meta = None
        self.resident = 0
        self.ok = False

    @classmethod
    def build(cls, sharded, spec, n_spec, w1, w2):
        """The steps object (`ok` False when it cannot serve, the same on every
        rank). Collective over the drafter TP group."""
        import os

        comm = sharded.comm
        dev = sharded.device
        steps = cls(sharded, n_spec)
        flag, why = 1, ""
        mode = os.environ.get(STEPS_MODE_ENV, "hoist").split(",")
        warps = int(os.environ.get(STEPS_WARPS_ENV, "4") or 4)
        try:
            if spec.use_fp64_gumbel:
                raise ValueError("FP64 Gumbel keys")
            if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
                raise ValueError("Markov weights not bf16")
            # One tile per CTA (its markov_w2 tile kept across the steps) when the
            # device holds that grid at once; "any" (tests) also tries fewer CTAs.
            want = int(os.environ.get(STEPS_CTAS_ENV, "0") or 0) or steps.tiles
            cands = sorted(
                (
                    c
                    for c in range(1, min(want, steps.tiles) + 1)
                    if steps.tiles % c == 0
                ),
                reverse=True,
            )
            if "any" not in mode:
                cands = [c for c in cands if c == steps.tiles]
            for ctas in cands:
                steps.ctas = ctas  # compiled for this count (a specialized value)
                meta = steps._meta_for(
                    w1,
                    ctas,
                    ctas == steps.tiles,
                    warps,
                )
                kernel = steps._compile(spec, w1, w2, meta)
                resident = _resident_ctas(kernel, dev)
                if resident >= ctas:
                    steps.ctas, steps.meta = ctas, meta
                    steps.kernel, steps.resident = kernel, resident
                    break
            if steps.meta is None:
                steps.ctas = 1
                raise ValueError(f"{steps.tiles} CTAs do not fit the device at once")
        except Exception as exc:  # compile or occupancy failure: stay on the launches
            flag, why = 0, f"{type(exc).__name__}: {exc}"
        steps.why = why
        # Every rank: the same flag and the same CTA count, else nobody uses it.
        t = torch.tensor([flag, steps.ctas, -steps.ctas], dtype=torch.int64, device=dev)
        torch.distributed.all_reduce(
            t, op=torch.distributed.ReduceOp.MIN, group=comm.group
        )
        steps.ok = bool(t[0].item() == 1 and t[1].item() == -t[2].item())
        torch.accelerator.synchronize(dev)
        torch.distributed.barrier(group=comm.group)
        if steps.ok:
            k = steps.kernel
            logger.info_once(
                "DSpark Markov sampling: %d draft steps in one launch over %d ranks "
                "(%d CTAs of %d warps, %d-entry tiles, %d resident, %d regs, "
                "%d B shared)",
                n_spec,
                steps.tp,
                steps.ctas,
                steps.meta["num_warps"],
                steps.vt,
                steps.resident,
                int(getattr(k, "n_regs", -1) or -1),
                int(k.metadata.shared),
            )
        else:
            logger.info_once(
                "DSpark Markov one-launch steps off: %s", why or "another rank"
            )
        return steps

    def serves(self, num_reqs, n_spec, local) -> bool:
        return (
            self.ok
            and n_spec == self.k
            and 1 <= num_reqs <= STEPS_RMAX
            and num_reqs * n_spec <= NOISE_ROWS
            and local.dtype == torch.bfloat16
            and local.is_contiguous()
            and local.shape[0] == num_reqs * n_spec
        )

    def _args(self, spec, num_reqs, base, out, w1, w2s, idx_map, sample_pos):
        tokens = spec.draft_tokens
        return (
            spec.input_buffers.input_ids,
            spec._anchor_idx,
            w1,
            w2s,
            base,
            base.stride(0),
            base.stride(1),
            out,
            out.stride(0),
            out.stride(1),
            idx_map,
            idx_map.stride(0),
            sample_pos,
            sample_pos.stride(0),
            spec.temperature,
            spec.seeds,
            self.noise,
            self.words,
            self.count,
            self.box,
            self.peers,
            tokens,
            tokens.stride(0),
            num_reqs,
            self.vs,
            self.voff,
            self.tiles,
            self.rank,
            self.ctas,
        )

    def _meta_for(self, w1, ctas, hoist, warps):
        return dict(
            K=self.k,
            W=self.tp,
            W_POW2=triton.next_power_of_2(self.tp),
            RANK=w1.shape[1],
            VT=self.vt,
            RB=RB,
            RMAX=STEPS_RMAX,
            SALT=_DRAFT_NOISE_SALT,
            CTAS_POW2=triton.next_power_of_2(ctas),
            RK_POW2=triton.next_power_of_2(STEPS_RMAX * self.k),
            HOIST=bool(hoist),
            num_warps=warps,
            num_stages=1,
            launch_cooperative_grid=True,
        )

    def _compile(self, spec, w1, w2, meta):
        """Compile (no launch) the kernel for `meta`."""
        n = self.k
        local = torch.empty(n, self.vs, dtype=torch.bfloat16, device=self.device)
        base = local.view(1, n, self.vs)
        out = torch.empty_like(local).view(1, n, self.vs)
        w2s = w2[self.voff : self.voff + self.vs]
        idx_map = spec.sample_idx_mapping[:n].view(1, n)
        sample_pos = spec.sample_pos[:n].view(1, n)
        args = self._args(spec, 1, base, out, w1, w2s, idx_map, sample_pos)
        return _markov_shard_steps_kernel.warmup(*args, grid=(self.ctas,), **meta)

    def run(self, spec, num_reqs, local, w1, w2):
        """`ShardedMarkov.run` (same tokens, same cache contents)."""
        n_spec, vs = self.k, self.vs
        base = local.view(num_reqs, n_spec, vs)
        final = torch.empty_like(local)
        out = final.view(num_reqs, n_spec, vs)
        w2s = w2[self.voff : self.voff + vs]
        idx_map = spec.sample_idx_mapping[: num_reqs * n_spec].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[: num_reqs * n_spec].view(num_reqs, n_spec)
        args = self._args(spec, num_reqs, base, out, w1, w2s, idx_map, sample_pos)
        kernel = _markov_shard_steps_kernel[(self.ctas,)](*args, **self.meta)
        if kernel is not self.kernel:
            logger.warning_once(
                "DSpark Markov one-launch steps: launched a kernel other than the "
                "one checked at build (cooperative launch still checks residency)"
            )
        full = self.comm.gather_logits(final)
        cache = spec.draft_logits.values
        _markov_cache_scatter_kernel[
            (num_reqs * n_spec, triton.cdiv(self.vocab, SCATTER_BLOCK))
        ](
            full,
            full.stride(0),
            cache,
            cache.stride(0),
            cache.stride(1),
            spec.draft_logits.row_mapping,
            idx_map,
            idx_map.stride(0),
            self.vocab,
            K=n_spec,
            BLOCK=SCATTER_BLOCK,
            num_warps=4,
        )
