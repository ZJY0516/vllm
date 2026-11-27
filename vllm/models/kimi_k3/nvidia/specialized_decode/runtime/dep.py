# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode at DEP (data-parallel attention at TP1 + expert-parallel
MoE; e.g. DEP16 / DEP32): every rank runs all 96 heads of its own requests, so the
attention needs no collective. The step is the stock layer loop (AttnRes, the EP
MoE with its dispatch / combine, KDA) with every K3.1 DSA layer's attention
specialized end to end for TP1:

  input projection   split-K (or cuBLAS) over the packed [q_a | kv_a | gate | wk |
                     wv] weight (14560 x 7168 BF16, 208 MB per layer: the whole
                     weight on one GPU), per batch (DEP_PROJECTION)
  core               PreparedDsa at 96 heads: the split query (mla.query_wide), the
                     fused FP8 indexer, FlashInfer on the query head-padded to 128
                     (the stock decode's kernel: trtllm-gen on fp8, prims_ts on
                     nvfp4_ds_mla), MlaValue
  o_proj             split-K (or cuBLAS) over the 7168 x 12288 weight (no
                     all-reduce at TP1), per batch (DEP_OPROJ)

Every KDA layer runs its input projection (split-K or cuBLAS, the 49376 x 7168
packed weight), the prepared KDA core at 96 heads (KdaSplit / KdaWide per step
kind and batch: conv, F_B gate, recurrence, gated norm; the RecoverSSM verify and
records where the deployment uses them) and o_proj (as the DSA layers'). On a
MegaMoE deployment every MoE layer runs the EpMoeLayer schedule (moe.ep.layer:
split-K [shared gate_up | latent down] front, router and shared expert on a side
stream beside the front and MegaMoE) up to its FRONT_MAX tokens.

Precision is the stock layer's at every stage (FP32 accumulation, BF16 outputs;
the split-K order differs). K3 MLA layers run stock (per-kind policy:
`DEP_KINDS`).

A linear whose weight is MXFP8 (E4M3 codes + E8M0 block scales; in the NVFP4
checkpoint, the KDA input projection and o_proj) runs the stock layer's MXFP8 GEMM
at every batch (its dynamic MXFP8 activation quantization and kernel, i.e. the
stock decode's numerics for that linear) instead of needing a BF16 copy of the
weight; BF16 linears are unchanged."""

import os
from typing import Any

import torch

from ..cutedsl.gemm.projection import PreparedProjection
from .providers import attention_kind, dsa_scratch, pack_dsa_projection

# Rows (tokens per DP rank: requests x (1 + draft tokens)) the DEP attention takes;
# every batch past it runs the stock layer (the dsa_scratch, prims_ts plans and the
# fused indexer's buffers are sized by it).
DEP_MAX_ROWS = 320
# The DEP attention's rows per KV cache dtype: up to its crossover against the stock
# layer (GB300, DSpark q5, per DSA layer incl. projection and o_proj, L2 flushed),
# both with the core head-padded to 128 (fi_heads) on the stock decode's kernel:
#   fp8 (trtllm-gen)  32K R1 / R4 / R16 / R32 / R64 1.41 / 1.36 / 1.26 / 1.20 /
#     1.10x; AgentX 1.39 / 1.34 / 1.39 / 1.38 / 1.36x
#   nvfp4_ds_mla (prims_ts)  32K R1 / R4 / R16 / R64 1.40 / 1.35 / 1.28 / 1.09x;
#     AgentX 1.37 / 1.34 / 1.36 / 1.33x
DEP_ROWS = dict.fromkeys(("fp8", "fp8_e4m3", "nvfp4_ds_mla"), DEP_MAX_ROWS)
# The attention layer kinds specialized at TP1. KDA at 96 heads (RecoverSSM
# verify + commit, FP32 state and conv, q5, us per layer stock / spec):
# R1 44.2 / 27.7, R4 82.6 / 53.5, R16 220 / 143, R64 792 / 481.
DEP_KINDS = ("dsa", "kda")


def _bins(table, batch):
    """The tactic of `batch` rows from a {max rows: tactic} table."""
    return table[next((n for n in sorted(table) if batch <= n), max(table))]


# Split-K tactics (M tile, N tile, split K, stages) per batch bin of the TP1 input
# projection (N 14560, K 7168: 208 MB) and o_proj (N 7168, K 12288: 176 MB); None:
# cuBLAS (torch.mm, the stock linear's GEMM). GB300, L2 flushed (write), us
# (SOL 25.5 / 21.5 at 8.18 TB/s):
#   projection   5 rows 36.3 (cuBLAS 43.4), 20: 36.9 (45.6), 80: 39.6 (45.5),
#                160: 43.8 (47.3), 320: 79.4 (two 160-row tiles stream the weight
#                twice; cuBLAS 57.0)
#   o_proj       5 rows 31.1 (stock 37.5), 20: 34.2 (39.7), 80: 39.4 (42.7),
#                160: 51.4 (43.6), 320: 57.6 (50.8)
DEP_PROJECTION = {
    8: (128, 8, 1, 6),
    16: (128, 16, 1, 6),
    32: (128, 32, 1, 5),
    64: (128, 64, 1, 4),
    128: (128, 128, 1, 3),
    160: (128, 160, 1, 3),
    DEP_MAX_ROWS: None,
}
# o_proj 17-64 rows: 64-row weight tiles (112 CTAs on 152 SMs) stream the weight
# at ~6.1-6.5 TB/s (GB300 cold L2, us: 20 rows 27.2 vs 30.6 at (128, 32, 1, 5);
# 60 rows 29.0 vs 35.2 at (128, 64, 1, 4)); past 64 rows the 64-token tile streams
# the weight twice and cuBLAS wins (80 rows 35.5 vs 37.6).
DEP_OPROJ = {
    8: (64, 8, 1, 8),
    16: (128, 16, 1, 6),
    32: (64, 32, 1, 8),
    64: (64, 64, 1, 6),
    DEP_MAX_ROWS: None,
}


def mxfp8_linear(linear) -> bool:
    """Whether `linear` holds an MXFP8 weight (E4M3 codes with E8M0 block scales)
    served by the stock MXFP8 linear kernel."""
    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and getattr(linear, "weight_scale", None) is not None
    )


def output_rows(linear) -> int:
    """The output features of a TP1 linear (an MXFP8 kernel may store its weight
    transposed, so not `weight.shape[0]`)."""
    if mxfp8_linear(linear):
        return linear.output_size_per_partition
    return linear.weight.shape[0]


def dep_policies(table, batches):
    """{batch: tactic} of the split-K batches of `table` (cuBLAS ones left out)."""
    tactics = {b: _bins(table, b) for b in batches}
    return {b: t for b, t in tactics.items() if t is not None}


class DepProjection:
    """A TP1 linear: split-K at the batches its table gives a tactic, else cuBLAS
    (`torch.mm` into the output: the stock linear's FP32-accumulated BF16 GEMM).
    An MXFP8 `linear` runs the stock layer's MXFP8 GEMM at every batch."""

    def __init__(self, weight, table, batches, linear=None):
        self.mx = linear if linear is not None and mxfp8_linear(linear) else None
        if self.mx is not None:
            self.weight, self.split, self.batches = None, None, set()
            return
        self.weight = weight
        policies = dep_policies(table, batches)
        self.split = (
            PreparedProjection(weight, policies).prepare() if policies else None
        )
        self.batches = set(policies)

    def run(self, x, out):
        if self.mx is not None:
            return self.mx.quant_method.apply(self.mx, x)
        if x.shape[0] in self.batches:
            assert self.split is not None
            return self.split.run(x, out)
        return torch.mm(x, self.weight.t(), out=out)


class DepAttention:
    """One K3.1 DSA layer's attention at TP1: the input projection, the prepared
    core and o_proj for `runtime.wide` rows (the DEP step rows)."""

    def __init__(self, decoder, runtime):
        attention = decoder.self_attn
        assert attention_kind(attention) == "dsa", "DEP specializes the DSA layers"
        self.layer, self.attention = decoder, attention
        weight = pack_dsa_projection(attention)
        batches = runtime.wide
        self.wide = batches
        self.projection = DepProjection(weight, DEP_PROJECTION, batches)
        o_weight = attention.o_proj.weight
        self.output = DepProjection(o_weight, DEP_OPROJ, batches, attention.o_proj)
        from ..cutedsl.dsa.prepared import PreparedDsa

        cap = max(batches)
        scratch = dsa_scratch(runtime, cap)
        policy = dict(queries=runtime.queries)
        policy.update(kernels=runtime.fi_kernels or None)
        policy.update(prims_ts_rows=runtime.prims_ts_rows or None)
        pitches = {b: weight.shape[0] for b in batches}
        self.core = PreparedDsa(attention, pitches, scratch, **policy)
        device = runtime.device
        empty = lambda *shape: torch.empty(  # noqa: E731
            shape, dtype=torch.bfloat16, device=device
        )
        if "dep.projected" not in runtime.buffers:
            runtime.buffers["dep.projected"] = empty(cap, weight.shape[0])
            runtime.buffers["dep.local"] = empty(cap, attention.num_local_heads * 128)
        self.projected = runtime.buffers["dep.projected"]
        self.local = runtime.buffers["dep.local"]

    def prepare_layout(self, caches):
        self.core.prepare_cache(caches[0])

    def run_wide(self, x, caches, metadata, slot_mapping, q):
        """The attention output [B, 7168] of the normalized rows `x`."""
        from vllm.forward_context import get_forward_context

        b = x.shape[0]
        projected = self.projection.run(x, self.projected[:b])
        local = self.local[:b]
        attention = self.attention
        slots = slot_mapping[attention.mla_attn.mla_attn.layer_name]
        index: Any = get_forward_context().attn_metadata[
            attention.dsa_indexer.k_cache.prefix
        ]
        self.core.run(
            projected,
            local,
            cache=caches[0],
            slots=slots,
            index_slots=index.slot_mapping,
            metadata=metadata,
            q=q,
        )
        return self.output.run(local, x.new_empty((b, x.shape[1])))


class DepKda:
    """One KDA layer's attention at TP1: the input projection, the prepared KDA
    core (96 heads) and o_proj for `runtime.wide` rows: q = 1 decode steps and
    uniform verify steps of `runtime.q` tokens per request."""

    def __init__(self, decoder, runtime):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        from ..cutedsl.kda.prepared import PreparedKda

        attention = decoder.self_attn
        assert attention_kind(attention) == "kda"
        self.layer, self.attention = decoder, attention
        linear = attention.in_proj_qkvgfab
        weight = linear.weight
        batches = runtime.wide
        self.wide = batches
        self.projection = DepProjection(weight, DEP_PROJECTION, batches, linear)
        self.output = DepProjection(
            attention.o_proj.weight, DEP_OPROJ, batches, attention.o_proj
        )
        pitch, q = output_rows(linear), runtime.q
        # Gate mode 0 (the F_B GEMM inside the core) at TP1; verify keys are
        # (requests, q), decode keys the batch.
        policies = {b: (pitch, 0) for b in batches}
        if q > 1:
            policies.update({(b // q, q): (pitch, 0) for b in batches if b % q == 0})
        self.core = PreparedKda(attention, policies, is_conv_state_dim_first())
        device, cap = runtime.device, max(batches)
        if "dep.kda.projected" not in runtime.buffers:
            runtime.buffers["dep.kda.projected"] = torch.empty(
                cap, pitch, dtype=torch.bfloat16, device=device
            )
            runtime.buffers["dep.kda.local"] = torch.empty(
                cap, self.core.dim, dtype=torch.bfloat16, device=device
            )
        self.projected = runtime.buffers["dep.kda.projected"]
        self.local = runtime.buffers["dep.kda.local"]

    def prepare_layout(self, caches):
        self.core.prepare_caches(caches)

    def run_wide(self, x, caches, metadata, slot_mapping, q):
        """The attention output [B, 7168] of the normalized rows `x`."""
        b = x.shape[0]
        projected = self.projection.run(x, self.projected[:b])
        local = self.local[:b]
        if q > 1:  # uniform q-token verify rows
            self.core.run(
                projected,
                local,
                caches=caches,
                slots=metadata.spec_state_indices_tensor,
                accepted=metadata.num_accepted_tokens,
                q=q,
                deferred=getattr(metadata, "recoverssm_deferred", None),
            )
        else:
            slots = metadata.non_spec_state_indices_tensor
            self.core.run(projected, local, caches=caches, slots=slots)
        return self.output.run(local, x.new_empty((b, x.shape[1])))


def _mx_shared(module) -> bool:
    """An MXFP8 shared expert (NVFP4 checkpoint) with a BF16 latent down."""
    se = module.shared_experts
    return (
        se is not None
        and mxfp8_linear(se.gate_up_proj)
        and mxfp8_linear(se.down_proj)
        and module.routed_expert_down_proj.weight.dtype == torch.bfloat16
    )


# Opt-in F1 / F3 (NVFP4 DEP decode MoE front). Both reproduce the
# stock numerics bit for bit (checked at 48-80 rows): F1 runs the router and the
# latent down as one split-K GEMM over the packed [router | latent down] BF16 weight
# with FP32 output (the router's own output dtype and split at these rows), then the
# latent columns get the one BF16 rounding the stock BF16-output GEMM applies; F3
# fuses the shared expert's SiTU-and-mul with the MXFP8 quantization of its output
# (FP32 SiTU with tanhf, one BF16 rounding, FlashInfer's MXFP8 recipe and swizzled
# 128x4 scale layout) and hands the codes to the down projection as a
# QuantizedActivation. No approximate instructions.
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _situ_mxfp8_kernel(gu_ptr, q_ptr, s_ptr, M, D: tl.constexpr, padded_sf_cols,
                       inv_e4m3_max, beta: tl.constexpr, linear_beta: tl.constexpr,
                       GROUPS: tl.constexpr):
    row = tl.program_id(0)
    gblk = tl.program_id(1)
    sf_col = gblk * GROUPS + tl.arange(0, GROUPS)
    sf_off = (sf_col % 4) + (sf_col // 4) * 512 + (row % 32) * 16 \
        + ((row % 128) // 32) * 4 + (row // 128) * (128 * padded_sf_cols)
    valid = row < M  # rows past M: the scale buffer's padding rows (zero scales)
    cols = gblk * GROUPS * 32 + tl.arange(0, GROUPS * 32)
    g = tl.load(gu_ptr + row * (2 * D) + cols, mask=valid & (cols < D),
                other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * (2 * D) + D + cols, mask=valid & (cols < D),
                other=0.0).to(tl.float32)
    gate = (0.5 * beta) * libdevice.tanh(g * (1.0 / beta)) \
        * (1.0 + libdevice.tanh(g * 0.5))
    up = linear_beta * libdevice.tanh(u * (1.0 / linear_beta))
    act = (gate * up).to(tl.bfloat16).to(tl.float32)
    a2 = tl.reshape(act, [GROUPS, 32])
    norm = tl.max(tl.abs(a2), axis=1) * inv_e4m3_max
    bits = norm.to(tl.int32, bitcast=True)
    exp_b = (bits >> 23) & 255
    mant = bits & 0x7FFFFF
    bump = tl.where(mant != 0, 1, 0)
    bump = tl.where((exp_b == 0) & (mant <= 0x400000), 0, bump)
    ue = tl.minimum(exp_b + bump, 254)
    ue = tl.where(norm <= 0.0, 0, ue)
    inv = tl.where(ue == 0, 0.0,
                   (tl.maximum(254 - ue, 0) << 23).to(tl.float32, bitcast=True))
    y = tl.maximum(tl.minimum(a2 * inv[:, None], 448.0), -448.0)
    q = tl.reshape(y, [GROUPS * 32]).to(tl.float8e4nv)
    tl.store(q_ptr + row * D + cols, q, mask=valid & (cols < D))
    tl.store(s_ptr + sf_off, tl.where(valid, ue, 0).to(tl.uint8))


def situ_mul_mxfp8(gate_up, beta, linear_beta, groups=8):
    """gate_up [M, 2D] BF16 -> (E4M3 [M, D], uint8 swizzled 128x4 scales)."""
    m, d = gate_up.shape[0], gate_up.shape[1] // 2
    padded_m = (m + 127) // 128 * 128
    sf_cols = d // 32
    q = torch.empty(m, d, dtype=torch.float8_e4m3fn, device=gate_up.device)
    s = torch.empty(padded_m * sf_cols, dtype=torch.uint8, device=gate_up.device)
    _situ_mxfp8_kernel[(padded_m, d // (32 * groups))](
        gate_up, q, s, m, d, sf_cols, 1.0 / 448.0, beta, linear_beta, groups,
        num_warps=4,
    )
    return q, s


@triton.jit
def _split_front_kernel(src_ptr, logits_ptr, latent_ptr, E: tl.constexpr,
                        L: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src_ptr + row * (E + L) + cols, mask=cols < E + L)
    tl.store(logits_ptr + row * E + cols, v, mask=cols < E)
    tl.store(latent_ptr + row * L + (cols - E), v.to(tl.bfloat16),
             mask=(cols >= E) & (cols < E + L))


def pack_router_latent(moe):
    """One [router | latent down] BF16 weight [896 + 3584, 7168]; the stock
    parameters become views of it (no extra memory)."""
    packed = getattr(moe, "f1_front_weight", None)
    if packed is not None:
        return packed
    gate, down = moe.gate.weight, moe.routed_expert_down_proj.weight
    packed = torch.cat([gate.data, down.data])
    gate.data = packed[: gate.shape[0]]
    down.data = packed[gate.shape[0]:]
    moe.f1_front_weight = packed
    return packed


_K31_PRE_STREAM = None
_K31_SIDE_STREAM = None


def _k31_pre_stream():
    """Pre-dispatch (M1): one stream for the early dispatch of every DepMoeMx layer."""
    global _K31_PRE_STREAM
    if _K31_PRE_STREAM is None:
        # High priority: its one CTA should take the first SM freed by the GEMMs it overlaps
        # (with K31_DEPMOE_SIDE_PRIO the side stream outranks -1, so take the device's highest)
        prio = -100 if os.environ.get("K31_DEPMOE_SIDE_PRIO") is not None else -1
        _K31_PRE_STREAM = torch.cuda.Stream(priority=prio)
    return _K31_PRE_STREAM


class DepMoeMx:
    """DepMoeMx (opt-in K31_NVFP4_DEPMOE=1): one MegaMoE layer of a DEP step whose shared
    expert is MXFP8, on the EpMoeLayer stream schedule with the front split in two:

      side  the shared expert from the layer start (its MXFP8 linears: dynamic MXFP8
            activation quantization and the stock GEMM; the stock SiTU-and-mul, or
            with K31_DEPMOE_F3=1 SiTU fused with the down projection's MXFP8 input
            quantization)
      main  router split-K (FP32 logits) + sigmoid top-16, latent down split-K (BF16)
            (with K31_DEPMOE_F1=1 one split-K over [router | latent down] above 32
            rows), the routed experts, the latent RMSNorm, then the up-projection
            accumulated into the shared output (cuBLAS addmm: one rounding, as stock)

    Precision is the stock module's (the split-K GEMMs as in EpMoeLayer); batches
    above FRONT_MAX run the stock module."""

    def __init__(self, module, batches, side_stream):
        from vllm.logger import init_logger
        from vllm.model_executor.layers.fusion.quant_activation import (
            get_input_quant_key,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp8Dynamic,
        )

        from ..cutedsl.moe.ep.layer import FRONT_MAX, front_tactic

        self.module = module
        self.side = side_stream or torch.cuda.Stream()
        # Opt-in: K31_DEPMOE_SIDE_PRIO=<p> runs the shared expert on its own stream at priority p
        # (negative = higher), so the side work finishes before MegaMoE takes the SMs.
        side_prio = os.environ.get("K31_DEPMOE_SIDE_PRIO")
        if side_prio is not None:
            global _K31_SIDE_STREAM
            if _K31_SIDE_STREAM is None:
                _K31_SIDE_STREAM = torch.cuda.Stream(priority=int(side_prio))
            self.side = _K31_SIDE_STREAM
        self.events = [torch.cuda.Event() for _ in range(4)]
        # Opt-in scheduling knobs: K31_DEPMOE_HIPRI=1 runs the main front (router / latent
        # down / top-k) on a high-priority stream so its GEMM gets SMs before the side
        # stream's; K31_DEPMOE_SIDE_AFTER=1 starts the side stream after the main front's
        # GEMM (event edge). Order of work and numerics are unchanged.
        self.hipri = os.environ.get("K31_DEPMOE_HIPRI") == "1"
        self.side_after = os.environ.get("K31_DEPMOE_SIDE_AFTER") == "1"
        self.hi = torch.cuda.Stream(priority=-10) if self.hipri else None
        fronts = [b for b in batches if b <= FRONT_MAX]
        # Opt-in: K31_DEPMOE_MAX=<rows> lets DepMoeMx also take the
        # batches above FRONT_MAX up to <rows> (unset: FRONT_MAX, unchanged). Their router
        # and latent down run per K31_DEPMOE_WIDE:
        #   stock  (default) the stock module's own gate (FP32 logits) and
        #          routed_expert_down_proj: the stock kernels on the same inputs, so the
        #          layer output equals the stock module's at these rows;
        #   splitk the split-K tactics of front_tactic (128-token tile, up to 128 rows;
        #          with K31_DEPMOE_F1=1 the packed F1 front), the same math as stock;
        #          rows above 128 (up to <rows>) take the stock kernels as in `stock`.
        # The shared expert (side stream), MegaMoE, RMSNorm and up addmm are unchanged.
        wide_max = int(os.environ.get("K31_DEPMOE_MAX", str(FRONT_MAX)))
        self.wide_mode = os.environ.get("K31_DEPMOE_WIDE", "stock")
        if self.wide_mode not in ("stock", "splitk"):
            raise ValueError(f"K31_DEPMOE_WIDE={self.wide_mode}: expected stock or splitk")
        wides = [b for b in batches if FRONT_MAX < b <= wide_max]
        split_max = 128 if self.wide_mode == "splitk" else FRONT_MAX  # front_tactic's widest token tile
        self.stock_rows = {b for b in wides if b > split_max}
        fronts = fronts + [b for b in wides if b <= split_max]
        gate_w, down_w = module.gate.weight, module.routed_expert_down_proj.weight
        f1 = (
            os.environ.get("K31_DEPMOE_F1") == "1"
            and gate_w.dtype == down_w.dtype == torch.bfloat16
            and gate_w.shape[1] == down_w.shape[1]
        )
        self.f1_rows = {b for b in fronts if b > 32} if f1 else set()
        if self.f1_rows:
            packed = pack_router_latent(module)
            self.front = PreparedProjection(
                packed,
                {b: front_tactic(b, split=2) for b in self.f1_rows},
                output_dtype=torch.float32,
            ).prepare()
            gate_w, down_w = module.gate.weight, module.routed_expert_down_proj.weight
        router = {b: front_tactic(b, split=4 if b <= 32 else 2) for b in fronts}
        down = {
            b: front_tactic(b, split=int(os.environ.get("K31_DEPMOE_DOWN_SPLIT", "2")))
            for b in fronts
        }
        self.router = PreparedProjection(
            gate_w, router, output_dtype=torch.float32
        ).prepare()
        self.down = PreparedProjection(down_w, down).prepare()
        self.batches = set(fronts) | self.stock_rows
        norm = module.routed_expert_norm
        self.norm_weight, self.eps = norm.weight, norm.variance_epsilon
        self.up = module.routed_expert_up_proj.weight
        self.latent = down_w.shape[0]
        self.experts_n = gate_w.shape[0]
        se = module.shared_experts
        act = se.act_fn
        self.f3 = (
            os.environ.get("K31_DEPMOE_F3") == "1"
            and type(act).__name__ == "SituAndMul"
            and getattr(act, "linear_beta", None) is not None
            and not getattr(se, "shard_sequence_parallel", False)
            and getattr(se, "gemm_rs_ar", None) is None
            and get_input_quant_key(se.down_proj) == kMxfp8Dynamic
            and se.down_proj.input_size_per_partition % 256 == 0
        )
        if self.f3:
            self.beta, self.linear_beta = float(act.beta), float(act.linear_beta)
            self.qkey = kMxfp8Dynamic
        # Opt-in (F3 CUDA variant, K31_F3=1): CUDA SiTU + MXFP8 quantization,
        # enabled only when its init probe is bitwise equal to the stock shared expert.
        self.f3v = None
        if os.environ.get("K31_F3") == "1":
            from vllm.model_executor.layers.fused_moe import k31_situ_mxfp8

            if k31_situ_mxfp8.shared_forward_supported(se):
                self.f3v = k31_situ_mxfp8.shared_forward
                self.f3 = False
        # Pre-dispatch (M1): routing staging + MegaMoE dispatch head on a third
        # stream right after the top-k (K31_M1=1 with the M1 DeepGEMM header)
        enabled = getattr(module.experts, "k31_m1_enabled", None)
        self.m1 = bool(enabled is not None and enabled())
        if self.m1:
            self.pre_stream = _k31_pre_stream()
            self.pre_events = [torch.cuda.Event() for _ in range(2)]
        if not getattr(DepMoeMx, "_logged", False):
            DepMoeMx._logged = True
            init_logger(__name__).info(
                "K31 DepMoeMx front: rows %s, wide %s (stock-op rows %s), F1 rows %s, F3 %s, "
                "F3v3 %s (stock act %s), hipri %s (priority %s), side_after %s, m1 %s, "
                "side priority %s",
                sorted(self.batches), self.wide_mode, sorted(self.stock_rows),
                sorted(self.f1_rows), self.f3, self.f3v is not None,
                getattr(getattr(act, "_forward_method", None), "__name__", "?"),
                self.hipri, self.hi.priority if self.hi is not None else None,
                self.side_after, (self.m1, self.pre_stream.priority if self.m1 else None),
                self.side.priority,
            )

    def serves(self, batch):
        return batch in self.batches

    def _shared(self, x):
        se = self.module.shared_experts
        if self.f3v is not None:
            return self.f3v(se, x)
        if not self.f3:
            return self.module.shared_experts(x)
        from vllm.model_executor.layers.fusion.quant_activation import (
            QuantizedActivation,
        )

        gate_up, _ = se.gate_up_proj(x)
        q, s = situ_mul_mxfp8(gate_up, self.beta, self.linear_beta)
        qa = QuantizedActivation(
            data=q, scale=s, orig_dtype=gate_up.dtype,
            orig_shape=torch.Size((gate_up.shape[0], q.shape[1])),
            quant_key=self.qkey,
        )
        out, _ = se.down_proj(qa)
        return out

    def run(self, x):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            fused_grouped_topk,
        )

        m, t = self.module, x.shape[0]
        if t not in self.batches:
            return m(x)
        main, ev = torch.cuda.current_stream(), self.events
        ev[0].record(main)
        front = self.hi if self.hipri else main
        if front is not main:
            front.wait_event(ev[0])
        if not self.side_after:
            self._launch_side(x, main, ev)
        e, lat = self.experts_n, self.latent
        stock_op = t in self.stock_rows
        with torch.cuda.stream(front):
            if not stock_op:
                logits = torch.empty(t, e, device=x.device, dtype=torch.float32)
                latent = torch.empty(t, lat, device=x.device, dtype=x.dtype)
            if stock_op:
                logits, _ = m.gate(x)
                if self.side_after:
                    ev[2].record(front)
                    self._launch_side(x, main, ev, after=ev[2])
            elif t in self.f1_rows:
                both = torch.empty(t, e + lat, device=x.device, dtype=torch.float32)
                self.front.run(x, both)
                if self.side_after:
                    ev[2].record(front)
                    self._launch_side(x, main, ev, after=ev[2])
                _split_front_kernel[(t, triton.cdiv(e + lat, 1024))](
                    both, logits, latent, e, lat, 1024, num_warps=4
                )
            else:
                self.router.run(x, logits)
                if self.side_after:
                    ev[2].record(front)
                    self._launch_side(x, main, ev, after=ev[2])
            topk_w, topk_ids = fused_grouped_topk(
                x, logits, m.experts.top_k, m.moe_renormalize,
                m.gate.e_score_correction_bias, 1, 1, "sigmoid", m.routed_scaling_factor,
            )
            if self.m1:
                ev_t, ev_p = self.pre_events
                ev_t.record(front)
                with torch.cuda.stream(self.pre_stream):
                    self.pre_stream.wait_event(ev_t)
                    topk_w.record_stream(self.pre_stream)
                    topk_ids.record_stream(self.pre_stream)
                    m.experts.k31_predispatch(topk_w, topk_ids)
                    ev_p.record(self.pre_stream)
                m.experts._k31_pre_event = ev_p
            if stock_op:
                latent, _ = m.routed_expert_down_proj(x)
            elif t not in self.f1_rows:
                self.down.run(x, latent)
        if front is not main:
            ev[3].record(front)
            main.wait_event(ev[3])
            for z in (latent, topk_w, topk_ids):
                z.record_stream(main)
        routed = m.experts(latent, topk_w, topk_ids)
        normed = torch.empty_like(routed)
        ops.rms_norm(normed, routed, self.norm_weight, self.eps)
        main.wait_event(ev[1])
        return self._shared_out.addmm_(normed, self.up.t())

    def _launch_side(self, x, main, ev, after=None):
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev[0] if after is None else after)
            shared = self._shared(x)
            # Consumed (and returned) on the main stream after ev[1].
            shared.record_stream(main)
            ev[1].record(self.side)
        self._shared_out = shared


def DepMoe(module, batches, side_stream):  # noqa: N802 (the stock name is a class)
    """The stock DepMoe, or DepMoeMx for an MXFP8 shared expert."""
    if _mx_shared(module):
        return DepMoeMx(module, batches, side_stream)
    return DepMoeBf16(module, batches, side_stream)


class DepMoeBf16:
    """One MegaMoE layer of a DEP step: the EpMoeLayer schedule over the stock
    module's weights (its batches up to FRONT_MAX; wider: the stock module)."""

    def __init__(self, module, batches, side_stream):
        from ..cutedsl.moe.ep.layer import EpMoeLayer, pack_front

        norm = module.routed_expert_norm
        weights = dict(
            gate=module.gate.weight,
            bias=module.gate.e_score_correction_bias,
            front=pack_front(module),
            shared_down=module.shared_experts.down_proj.weight,
            norm=norm.weight,
            up=module.routed_expert_up_proj.weight,
            eps=norm.variance_epsilon,
            renormalize=module.moe_renormalize,
            scale=module.routed_scaling_factor,
        )
        experts = module.experts

        def routed(latent, topk_weights, topk_ids):
            return experts(latent, topk_weights, topk_ids)

        act = module.shared_experts.act_fn
        self.layer = EpMoeLayer(weights, act, routed, batches, side_stream=side_stream)

    def serves(self, batch):
        return batch in self.layer.batches

    def run(self, x):
        return self.layer(x)


def dep_moe_supported(module) -> bool:
    """Whether a DEP MoE layer runs DepMoe: a MegaMoE latent MoE with a shared
    expert and a latent RMSNorm, routed by one-group sigmoid top-16 (EpMoeLayer.topk;
    else stock)."""
    return (
        getattr(module, "use_mega_moe", False)
        and getattr(module.experts, "top_k", None) == 16
        and module.shared_experts is not None
        and module.routed_expert_norm is not None
        and module.use_grouped_topk
        and module.num_expert_group == 1
        and module.topk_group == 1
        and module.moe_router_activation_func == "sigmoid"
        # EpMoeLayer packs a BF16 [shared gate_up | latent down] front;
        # an NVFP4 checkpoint (MXFP8 shared expert) runs DepMoeMx with
        # K31_NVFP4_DEPMOE=1, else the stock MoE module.
        and (
            (
                module.shared_experts.gate_up_proj.weight.dtype == torch.bfloat16
                and module.routed_expert_down_proj.weight.dtype == torch.bfloat16
            )
            or (os.environ.get("K31_NVFP4_DEPMOE") == "1" and _mx_shared(module))
        )
    )
