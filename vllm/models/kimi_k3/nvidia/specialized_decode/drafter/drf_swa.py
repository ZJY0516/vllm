# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch decode attention for the DSpark draft layers (K31_DRF).

The draft layer's attention is, per step: q RMSNorm, k RMSNorm, RoPE, the K/V
cache write of the query rows, and a paged sliding-window attention of those
rows (FA4: scheduler prepare, split forward, combine). For a draft step the
query block is a handful of rows per request (the speculative block), the
window is short (2048) and the head dim is 64, so each of those launches is a
fixed cost. `DraftSwaAttention` runs all of it as one Triton launch:

* grid (split, head group, request). Every program owns `BLOCK_N * TILES`
  cache keys of the request's window (positions before the query block) for
  `HPC` query heads of one KV head (GQA rows packed as `HPC * N` matrix rows);
  split 0 also owns the query block's own keys, computed in registers from the
  QKV projection (k RMSNorm + RoPE; v as is) instead of read back from the cache.
* the first cache tile is loaded before the PDL wait (it does not depend on the
  QKV projection that precedes this launch).
* the q/k RMSNorm and RoPE use the float formulas of vLLM's `rms_norm` and
  `rotary_embedding` kernels (one BF16 rounding each); the query rows' K/V are
  written to the cache at `slot_mapping` as `reshape_and_cache_flash` does.
* each program writes its unnormalized partial (O, row max, row sum) and the
  last program of a (request, head group) to arrive merges the splits (an
  arrival counter that the merging program resets, so CUDA-graph replays need
  no host work) and writes the BF16 output.

Softmax: base-2 online softmax, P rounded to BF16 for the PV product, FP32
accumulation (as FA). Only the split boundaries and the merge order differ from
FA4, so the output is not bit-identical to it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:  # Triton PDL intrinsics (Triton >= 3.4).
    _gdc_wait = tl.extra.cuda.gdc_wait
    _gdc_launch = tl.extra.cuda.gdc_launch_dependents
    HAS_PDL = True
except AttributeError:  # pragma: no cover
    HAS_PDL = False

_LOG2E = 1.4426950408889634


@triton.jit
def _norm_rope(
    src_ptr, tok, col, ok, pos, w_ptr, cs_ptr, cs_stride, eps,
    offs_d, D: tl.constexpr, HALF: tl.constexpr, src_stride,
):
    """RMSNorm (weight w) then neox RoPE of the [rows, D] block at
    src[tok, col:col+D]; returns BF16. Float math, one rounding per op as the
    vLLM kernels (x * inv * w -> BF16; x*cos -/+ y*sin -> BF16)."""
    partner = (offs_d + HALF) % D
    row = tok.to(tl.int64) * src_stride + col
    x = tl.load(src_ptr + row[:, None] + offs_d[None, :], mask=ok[:, None], other=0.0)
    xp = tl.load(src_ptr + row[:, None] + partner[None, :], mask=ok[:, None], other=0.0)
    x = x.to(tl.float32)
    xp = xp.to(tl.float32)
    var = tl.sum(x * x, axis=1) * (1.0 / D)
    inv = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    wp = tl.load(w_ptr + partner).to(tl.float32)
    xn = (x * inv[:, None] * w[None, :]).to(tl.bfloat16).to(tl.float32)
    xpn = (xp * inv[:, None] * wp[None, :]).to(tl.bfloat16).to(tl.float32)
    cidx = offs_d % HALF
    cs_row = pos.to(tl.int64) * cs_stride
    cos = tl.load(cs_ptr + cs_row[:, None] + cidx[None, :], mask=ok[:, None], other=0.0)
    sin = tl.load(
        cs_ptr + cs_row[:, None] + HALF + cidx[None, :], mask=ok[:, None], other=0.0
    )
    cos = cos.to(tl.float32)
    sin = sin.to(tl.float32)
    first = (offs_d < HALF)[None, :]
    # d < HALF: x*cos - y*sin (y = partner); d >= HALF: y*cos + x*sin (x = partner)
    out = tl.where(first, xn * cos - xpn * sin, xn * cos + xpn * sin)
    return out.to(tl.bfloat16)


@triton.jit
def _drf_swa_attn_kernel(
    qkv_ptr, qkv_stride,
    pos_ptr, qw_ptr, kw_ptr, eps,
    cs_ptr, cs_stride, max_pos,
    kv_ptr, kv_sb, kv_sh, kv_st,
    bt_ptr, bt_stride, sl_ptr, qsl_ptr, slot_ptr,
    out_ptr, out_stride,
    po_ptr, pml_ptr, cnt_ptr,
    qk_scale_log2,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
    NP: tl.constexpr, HPC: tl.constexpr, M: tl.constexpr, PAGE: tl.constexpr,
    WINDOW: tl.constexpr, BLOCK_N: tl.constexpr, TILES: tl.constexpr,
    NSPLIT: tl.constexpr, USE_PDL: tl.constexpr, WRITE_CACHE: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    s = tl.program_id(0)
    hg = tl.program_id(1)
    r = tl.program_id(2)
    NHG: tl.constexpr = HQ // HPC
    G: tl.constexpr = HQ // HKV
    HALF: tl.constexpr = D // 2
    SPLIT: tl.constexpr = BLOCK_N * TILES
    h0 = hg * HPC
    kvh = h0 // G
    if USE_PDL and EARLY_TRIGGER:
        _gdc_launch()

    q_start = tl.load(qsl_ptr + r)
    seq_len = tl.load(sl_ptr + r)
    p_first = seq_len - N  # first query position (query rows are the last N)
    lo = tl.maximum(p_first - (WINDOW - 1), 0)
    k_begin = lo + s * SPLIT

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_m = tl.arange(0, M)
    row_h = offs_m // N
    row_i = offs_m % N
    row_ok = offs_m < HPC * N
    kv_head = kvh.to(tl.int64) * kv_sh
    bt_row = bt_ptr + r.to(tl.int64) * bt_stride

    # First cache tile before the PDL wait (independent of the QKV projection).
    kp = k_begin + offs_n
    kok = kp < p_first
    blk = tl.load(bt_row + kp // PAGE, mask=kok, other=0)
    base = blk.to(tl.int64) * kv_sb + kv_head + (kp % PAGE).to(tl.int64) * kv_st
    k_t = tl.load(kv_ptr + base[:, None] + offs_d[None, :], mask=kok[:, None], other=0.0)
    v_t = tl.load(
        kv_ptr + base[:, None] + D + offs_d[None, :], mask=kok[:, None], other=0.0
    )
    tok = q_start + row_i
    pos = tl.load(pos_ptr + tok, mask=row_ok, other=0)
    pos = tl.minimum(tl.maximum(pos, 0), max_pos - 1)

    if USE_PDL:
        _gdc_wait()

    q = _norm_rope(
        qkv_ptr, tok, (h0 + row_h) * D, row_ok, pos, qw_ptr, cs_ptr, cs_stride, eps,
        offs_d, D, HALF, qkv_stride,
    )
    # FA aligns the query rows to the end of the keys: row i sits at p_first + i.
    lo_row = p_first + row_i - (WINDOW - 1)

    m_i = tl.full([M], float("-inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, D], tl.float32)
    for t in tl.static_range(TILES):
        if t > 0:
            kp = k_begin + t * BLOCK_N + offs_n
            kok = kp < p_first
            blk = tl.load(bt_row + kp // PAGE, mask=kok, other=0)
            base = blk.to(tl.int64) * kv_sb + kv_head + (kp % PAGE).to(tl.int64) * kv_st
            k_t = tl.load(
                kv_ptr + base[:, None] + offs_d[None, :], mask=kok[:, None], other=0.0
            )
            v_t = tl.load(
                kv_ptr + base[:, None] + D + offs_d[None, :], mask=kok[:, None],
                other=0.0,
            )
        sc = tl.dot(q, tl.trans(k_t))
        valid = kok[None, :] & (kp[None, :] >= lo_row[:, None]) & row_ok[:, None]
        sc = tl.where(valid, sc * qk_scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_t)
        m_i = m_new

    if s == 0:
        # The query block's own keys: k RMSNorm + RoPE, v as is, from the QKV rows.
        offs_j = tl.arange(0, NP)
        jok = offs_j < N
        ktok = q_start + offs_j
        kpos = tl.load(pos_ptr + ktok, mask=jok, other=0)
        kpos = tl.minimum(tl.maximum(kpos, 0), max_pos - 1)
        k_new = _norm_rope(
            qkv_ptr, ktok, (HQ + kvh) * D, jok, kpos, kw_ptr, cs_ptr, cs_stride, eps,
            offs_d, D, HALF, qkv_stride,
        )
        vrow = ktok.to(tl.int64) * qkv_stride + (HQ + HKV + kvh) * D
        v_new = tl.load(
            qkv_ptr + vrow[:, None] + offs_d[None, :], mask=jok[:, None], other=0.0
        )
        sc_n = tl.dot(q, tl.trans(k_new))
        valid_n = jok[None, :] & (offs_j[None, :] <= row_i[:, None]) & row_ok[:, None]
        sc_n = tl.where(valid_n, sc_n * qk_scale_log2, float("-inf"))
        mn_n = tl.maximum(m_i, tl.max(sc_n, axis=1))
        ms_n = tl.where(mn_n == float("-inf"), 0.0, mn_n)
        al_n = tl.exp2(m_i - ms_n)
        p_n = tl.exp2(sc_n - ms_n[:, None])
        l_i = l_i * al_n + tl.sum(p_n, axis=1)
        acc = acc * al_n[:, None] + tl.dot(p_n.to(tl.bfloat16), v_new)
        m_i = mn_n
        if WRITE_CACHE:
            if h0 % G == 0:
                slot = tl.load(slot_ptr + ktok, mask=jok, other=-1)
                wok = jok & (slot >= 0)
                wbase = (slot // PAGE) * kv_sb + kv_head + (slot % PAGE) * kv_st
                tl.store(
                    kv_ptr + wbase[:, None] + offs_d[None, :], k_new, mask=wok[:, None]
                )
                tl.store(
                    kv_ptr + wbase[:, None] + D + offs_d[None, :], v_new,
                    mask=wok[:, None],
                )

    grp = r * NHG + hg
    pidx = (grp * NSPLIT + s).to(tl.int64)
    tl.store(po_ptr + pidx * (M * D) + offs_m[:, None] * D + offs_d[None, :], acc)
    tl.store(pml_ptr + pidx * (2 * M) + offs_m, m_i)
    tl.store(pml_ptr + pidx * (2 * M) + M + offs_m, l_i)
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr + grp, 1, sem="acq_rel", scope="gpu")
    if arrived == NSPLIT - 1:
        tl.debug_barrier()
        gbase = (grp * NSPLIT).to(tl.int64)
        m_g = tl.full([M], float("-inf"), tl.float32)
        for s2 in tl.static_range(NSPLIT):
            ms = tl.load(pml_ptr + (gbase + s2) * (2 * M) + offs_m, cache_modifier=".cg")
            m_g = tl.maximum(m_g, ms)
        m_gs = tl.where(m_g == float("-inf"), 0.0, m_g)
        o = tl.zeros([M, D], tl.float32)
        lsum = tl.zeros([M], tl.float32)
        for s2 in tl.static_range(NSPLIT):
            pb = gbase + s2
            ms = tl.load(pml_ptr + pb * (2 * M) + offs_m, cache_modifier=".cg")
            ls = tl.load(pml_ptr + pb * (2 * M) + M + offs_m, cache_modifier=".cg")
            os_ = tl.load(
                po_ptr + pb * (M * D) + offs_m[:, None] * D + offs_d[None, :],
                cache_modifier=".cg",
            )
            sc2 = tl.exp2(ms - m_gs)
            lsum += ls * sc2
            o += os_ * sc2[:, None]
        o = o / lsum[:, None]
        orow = tok.to(tl.int64) * out_stride + (h0 + row_h) * D
        tl.store(
            out_ptr + orow[:, None] + offs_d[None, :], o.to(tl.bfloat16),
            mask=row_ok[:, None],
        )
        tl.store(cnt_ptr + grp, 0)
    if USE_PDL and not EARLY_TRIGGER:
        _gdc_launch()


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


class DraftSwaAttention:
    """Launcher of one draft layer shape. The scratch (partials and arrival
    counters, sized for `max_reqs` requests of at most 16 packed rows per
    program) is allocated here, outside any CUDA-graph capture, and may be shared
    by every layer of a model (`shared`): the layers run one after another on one
    stream, and each merge leaves its counters at zero."""

    MAX_M = 16

    def __init__(
        self, num_heads: int, num_kv_heads: int, head_dim: int, window: int,
        eps: float, scale: float, device, *, max_reqs: int = 1,
        block_n: int = 256, tiles: int = 1, hpc: int | None = None,
        use_pdl: bool = True, early_trigger: bool = True, write_cache: bool = True,
        num_warps: int = 4, num_stages: int = 1,
        shared: "DraftSwaAttention | None" = None,
    ) -> None:
        self.hq, self.hkv, self.d, self.window = num_heads, num_kv_heads, head_dim, window
        self.eps, self.scale = float(eps), float(scale)
        self.block_n, self.tiles = block_n, tiles
        self.hpc = hpc
        self.use_pdl = bool(use_pdl and HAS_PDL)
        self.early_trigger = early_trigger
        self.write_cache = write_cache
        self.num_warps, self.num_stages = num_warps, num_stages
        self.max_reqs = max_reqs
        split = block_n * tiles
        self.nsplit = -(-(window - 1) // split)
        if shared is not None:
            assert shared.nsplit == self.nsplit and shared.hq == self.hq
            self.po, self.pml, self.cnt = shared.po, shared.pml, shared.cnt
            self.max_reqs = shared.max_reqs
        else:
            groups = max_reqs * num_heads
            m = self.MAX_M
            self.po = torch.empty(groups * self.nsplit * m * head_dim,
                                  dtype=torch.float32, device=device)
            self.pml = torch.empty(groups * self.nsplit * 2 * m, dtype=torch.float32,
                                   device=device)
            self.cnt = torch.zeros(groups, dtype=torch.int32, device=device)

    def config(self, n: int):
        g = self.hq // self.hkv
        hpc = self.hpc
        if hpc is None:  # heads per program: fill 16 MMA rows
            hpc = 1
            for c in range(1, g + 1):
                if g % c == 0 and c * n <= 16:
                    hpc = c
        m = max(16, _next_pow2(hpc * n))
        np_ = max(16, _next_pow2(n))
        return hpc, m, np_, self.nsplit

    def supports(self, num_reqs: int, n: int) -> bool:
        hpc, m, np_, _ = self.config(n)
        return m <= self.MAX_M and np_ <= 16 and 1 <= num_reqs <= self.max_reqs

    def __call__(self, qkv, positions, q_w, k_w, cos_sin, kv_cache, block_table,
                 seq_lens, query_start_loc, slot_mapping, out, num_reqs: int,
                 n: int):
        hpc, m, np_, nsplit = self.config(n)
        assert self.supports(num_reqs, n)
        # kv_cache [num_blocks, num_kv_heads, page, 2 * head_dim]
        sb, sh, st, sd = kv_cache.stride()
        assert sd == 1 and kv_cache.shape[-1] == 2 * self.d
        grid = (nsplit, self.hq // hpc, num_reqs)
        _drf_swa_attn_kernel[grid](
            qkv, qkv.stride(0), positions, q_w, k_w, self.eps, cos_sin,
            cos_sin.stride(0), cos_sin.shape[0], kv_cache, sb, sh, st, block_table,
            block_table.stride(0), seq_lens, query_start_loc, slot_mapping, out,
            out.stride(0), self.po, self.pml, self.cnt, self.scale * _LOG2E,
            HQ=self.hq, HKV=self.hkv, D=self.d, N=n, NP=np_, HPC=hpc, M=m,
            PAGE=kv_cache.shape[2], WINDOW=self.window, BLOCK_N=self.block_n,
            TILES=self.tiles, NSPLIT=nsplit, USE_PDL=self.use_pdl,
            WRITE_CACHE=self.write_cache, EARLY_TRIGGER=self.early_trigger,
            num_warps=self.num_warps, num_stages=self.num_stages,
            launch_pdl=self.use_pdl,
        )
        return out
