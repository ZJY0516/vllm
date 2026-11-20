# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed MoE tail: one fused kernel finalizes this rank's top-16 rows, multicasts the
latent partial and peer-stores the shared shard (three rotating parities), all-reduces
and RMS-normalizes per K-slice, runs the K-split up projection and multicasts the output
row into the next AttnRes mailbox. At TP8 B5+ and TP16 B9+ the experts publish the
latent partial, WaveMoeTail.push the shared shard and WaveMoeTail does the rest (TP16
B5-8 under `PUB8`: WaveMoeTailExact, with the front tail's arithmetic). At
`twoshot_batches` the latent all-reduce is `common.twoshot`: the experts push each
partial chunk to its column owner, whose WaveMoeTail CTAs sum and multicast it."""

import os
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, prepare_shared_carveout, stream, tensor_maps, to_cute
from ..common.policy import STEP_CAP, W112, batch_table
from ..common.twoshot import TwoShot
from ..oproj.prepared import OprojMailbox
from .tail import MoeTail, WaveMoeTail, WaveMoeTailExact

# MoE tail mailbox tokens (padded decode batches); small-batch front_poll G/W/S rows per
# parity.
CAP, POLL_CAP = STEP_CAP, 8
# Two-shot latent all-reduce batches per TP (B9+): 2-/4-node GB300, us/layer of RS push
# + experts + WaveMoeTail, max over ranks, multicast vs two-shot: TP8 B16 41.74/42.80,
# B32 52.20/53.18, B64 79.27/81.50; TP16 B16 33.98/34.49, B32 42.19/42.52, B64
# 65.05/60.51.
TWOSHOT = {8: "", 16: "33-64:1"}
# WaveMoeTail batches that release the next AttnRes at entry (every published batch:
# TP8 B5+, TP16 B9+). The AttnRes then runs its saved-block statistics and exchange
# beside the tail and only its Lamport poll waits for the output rows; the next input
# projection's weight prologue starts with it. GB300 TP8 K3.1 DSpark k=4 step, R1 (B5),
# ms, release after the output multicast -> at entry: DCP8 136.5K 10.26 -> 10.21.
EARLY = {8: "1-64:1", 16: "9-64:1"}
# WaveMoeTail batches whose up projection runs on tcgen05 (B9+; 1-GPU WaveMoeTail us
# mma.sync vs tcgen05, B16/32/64: TP8 4.53/5.80/9.05 vs 4.45/5.26/7.80, TP16 two-shot
# 4.50/5.39/7.75 vs 4.52/5.46/7.21).
UMMA = {8: "9-64:1", 16: "33-64:1"}
# K31_W112: the TP8 full step's 65-112 rows run the published tcgen05 tail too,
# releasing the next AttnRes at entry.
if W112:
    UMMA[8], EARLY[8] = "9-112:1", "1-112:1"
# Front-mode MoeTail batches per (TP size, DCP size). TAIL_GATE: one CTA barrier
# between the CTA's front tasks (routed finalize, shared-shard stores) and its
# latent/RS poll. POLL_NS: the latent/RS poll's sleep between rounds, ns (absent:
# MoeTail's default). The tail reads the same values and sums them in the same order:
# the same output bits. TP16 + DCP16 B5-8 (DSpark verify steps of one request, k =
# 4..7): faster per layer and per decode step than the defaults in A/B. Other (TP, DCP)
# pairs and batches: the defaults.
TAIL_GATE = {(16, 16): "5-8:1"}
POLL_NS = {(16, 16): "5-8:0"}
# TP16 verify batches published by the experts and finished by WaveMoeTailExact
# (same output bits as the front tail); with them the MoE front gates the shared
# expert (moe.front.EARLY_ROUTE). Default off; K31_PUB8=1 enables B5-8.
# K31_PUB8_POLL0=1: its latent/RS polls retry without sleeping (as POLL_NS does for
# the front tail); K31_PUB8_EARLY=1: it releases the next AttnRes at entry (EARLY).
# Both change only timing, never what is loaded, summed or stored.
_on = lambda name: os.environ.get(name, "0") == "1"
PUB8 = {16: "5-8:1" if _on("K31_PUB8") else ""}
PUB8_POLL_NS = 0 if _on("K31_PUB8_POLL0") else None
# K31_PUB8_RSWARP=1: a ninth warp polls its RS rows from entry (same sums).
PUB8_RS_WARP = _on("K31_PUB8_RSWARP")
if _on("K31_PUB8_EARLY"):
    EARLY[16] += ",5-8:1"
# TP8 WaveMoeTail options (all default off; none changes what is loaded, summed or
# stored, so the output bits are the same). K31_T8T=1: past one group (9-16 rows,
# two or three requests at k = 4), every CTA polls its slice of each token from the
# sources itself instead of reading latent rows another cluster reduced
# (WaveMoeTail.direct). K31_T8T_POLL0=1: the published tails' polls (5-16 rows)
# retry without sleeping. K31_T8T_VOTE=1: the published tails' polls (5-16 rows) run
# on whole warps, each retry pass reloading every source, and a warp vote ends them.
T8T = {8: "9-16:1" if _on("K31_T8T") else ""}
T8T_POLL0 = {8: "5-16:1" if _on("K31_T8T_POLL0") else ""}
T8T_VOTE = {8: "5-16:1" if _on("K31_T8T_VOTE") else ""}
# K31_S21T8T=1: the POLL0 and VOTE entries above (when on) reach 32 rows (three or
# four groups at k = 4); K31_S21T8D=1: the T8T entry (when on) too. Same code paths,
# wider tables.
if _on("K31_S21T8T"):
    for _table in (T8T_POLL0, T8T_VOTE):
        if _table[8]:
            _table[8] = "5-32:1"
if _on("K31_S21T8D") and T8T[8]:
    T8T[8] = "9-32:1"


# K31_S23DR=1: TP8 published tcgen05 tails of 17-40 rows start the up projection
# before the RMS exchange completes and scale each output line by its token's inverse
# RMS (WaveMoeTail.late; not the same bits as normalize-first). Default off.
LATE = {8: "17-40:1"} if _on("K31_S23DR") else {}


def late(tp, batch):
    return bool(batch_table(LATE.get(tp, "")).get(batch, 0))


def flat(t):
    return to_cute(t.reshape(-1), 16)


def pub8(tp, batch):
    """TP16 B<=8 batch on the published path (`PUB8`), run by WaveMoeTailExact."""
    table = batch_table(PUB8.get(tp, ""))
    return tp == 16 and batch <= 8 and bool(table.get(batch, 0))


def published(tp, batch):
    """The experts publish the latent and WaveMoeTail runs the tail (else MoeTail
    front)."""
    return batch > (4 if tp == 8 else 8) or pub8(tp, batch)


def direct(tp, batch, twoshot=False):
    """TP8 published batch past one group whose tail polls directly (`T8T`)."""
    table = batch_table(T8T.get(tp, ""))
    on = tp == 8 and batch > 8 and not twoshot and bool(table.get(batch, 0))
    return on and published(tp, batch)


def published_vote(tp, batch, twoshot=False):
    """TP8 published batch whose polls end by a warp vote (`T8T_VOTE`)."""
    table = batch_table(T8T_VOTE.get(tp, ""))
    on = tp == 8 and not twoshot and bool(table.get(batch, 0))
    return on and published(tp, batch)


def published_poll_ns(tp, batch):
    """TP8 published batch's poll sleep in ns (`T8T_POLL0`: 0), or None."""
    table = batch_table(T8T_POLL0.get(tp, ""))
    on = tp == 8 and published(tp, batch) and bool(table.get(batch, 0))
    return 0 if on else None


def front_gate(tp, dcp, batch):
    """MoeTail front batch with the barrier before the latent/RS poll (TAIL_GATE)."""
    table = batch_table(TAIL_GATE.get((tp, dcp), ""))
    return not published(tp, batch) and bool(table.get(batch, 0))


def front_poll_ns(tp, dcp, batch):
    """MoeTail front batch's latent/RS poll sleep in ns (POLL_NS), or None."""
    table = batch_table(POLL_NS.get((tp, dcp), ""))
    return None if published(tp, batch) else table.get(batch)


def twoshot_batches(tp):
    table = batch_table(TWOSHOT.get(tp, ""))
    return {batch for batch, on in table.items() if on and batch > 8}


def early_batches(tp):
    table = batch_table(EARLY.get(tp, ""))
    return {batch for batch, on in table.items() if on and published(tp, batch)}


def umma_batches(tp):
    table = batch_table(UMMA.get(tp, ""))
    return {batch for batch, on in table.items() if on and batch > 8}


def shares(tp, batch, twoshot=False):
    """WaveMoeTail clusters share reduced latent rows (all 112 CTAs must be
    co-resident)."""
    if pub8(tp, batch):
        return False
    return published(tp, batch) and WaveMoeTail(batch, tp, twoshot).reduce


@cache
def resident_clusters(device):
    """Co-resident 16-CTA clusters at one CTA per SM."""
    with torch.accelerator.device_index(device):
        return cutlass.utils.HardwareInfo(device).get_max_active_clusters(16)


class MoeTailBox:
    """Model-owned MoE tail mailboxes: latent [3, TP, CAP, 3584] (multicast), shared
    reduce-scatter [3, TP, CAP, 7168/TP] (peer stores), output [1, CAP, 7168], the
    local all-reduced latent rows [3, CAP, 3584] that WaveMoeTail clusters share and the
    two-shot latent buffers (`twoshot_args`/`twoshot_tail` replace args/tail)."""

    def __init__(self, group, device, tp, rank):
        self.rank = rank
        self.latent = OprojMailbox(group, device, tp, rank, (3, tp, CAP, 3584))
        self.rs = OprojMailbox(  # peer stores only
            group, device, tp, rank, (3, tp, CAP, 7168 // tp), multicast=False
        )
        self.out = OprojMailbox(group, device, tp, rank, (1, CAP, 7168))
        ptrs = self.rs.handle.buffer_ptrs
        self.peers = torch.tensor(ptrs, dtype=torch.int64, device=device)
        self.args = (flat(self.latent.buffer), flat(self.rs.buffer))
        self.tail: tuple = (to_cute(self.peers, 16), flat(self.out.buffer))
        self.tail += (Int64(self.latent.mc), Int64(self.out.mc))
        self.reduced = torch.empty((3, CAP, 3584), dtype=torch.bfloat16, device=device)
        self.reduced.view(torch.int32).fill_(-0x80000000)
        self.twoshot = box = TwoShot(group, device, tp, rank, 3584, CAP, parities=3)
        self.twoshot_args = (flat(box.recv), self.args[1])
        self.twoshot_tail = (*self.tail[:2], Int64(box.gather_mc), self.tail[3])
        # Experts' push: every parity's receive slots [3 x TP, CAP, 3584/TP] and the
        # owners' buffer bases (slot parity x TP + rank).
        self.push_box = (
            to_cute(box.recv.view(3 * tp, CAP, -1)),
            to_cute(box.peers[0], 16),
        )


@cache
def compile_tail(
    tp,
    batch,
    device,
    mode="front",
    trigger=2,
    early=False,
    umma=False,
    gate=False,
    poll_ns=None,
    direct=False,
    vote=False,
    late=False,
):
    """front: rows complete at the PDL wait; front_poll: three 0xffff-armed G/W/S parity
    slots polled per row and re-armed after use; publish (`published`): WaveMoeTail
    over the published mailboxes, reading the weight shard directly, not by TMA;
    twoshot: publish over the two-shot receive slots and gather. `gate` and
    `poll_ns`: MoeTail's barrier before the latent/RS poll and its poll sleep
    (`front_gate`, `front_poll_ns`; published: `published_poll_ns`). `direct`,
    `vote`: WaveMoeTail's (`T8T`)."""
    m8, twoshot = mode in ("publish", "twoshot"), mode == "twoshot"
    vec = lambda n, dtype=BFloat16, align=16: fake((n,), None, dtype, align)
    slots = 3 * POLL_CAP if mode == "front_poll" else batch
    latent = 3 * tp * CAP * (3584 // tp if twoshot else 3584)
    tensors: tuple = (vec(latent), vec(3 * tp * CAP * (7168 // tp)))
    weights = vec(7168 // tp * 3584) if m8 and not umma else vec(128, Uint8, 64)
    tensors += (weights, vec(3584))
    tensors += (vec(slots * 16 * 3584), vec(slots * 16), vec(slots * 7168))
    tensors += (vec(tp, Int64), vec(CAP * 7168))
    scalars: tuple = (Int64(0), Int64(0), Int32(0), Int32(0), Int32(CAP), Float32(1e-6))
    scalars += (Int32(2), Int32(CAP))
    kernel: MoeTail
    if m8 and pub8(tp, batch):
        kernel = wave = WaveMoeTailExact(batch, tp, early, poll_ns, PUB8_RS_WARP)
    elif m8 and (direct or vote or poll_ns is not None):
        kw = dict(direct=direct, poll_ns=poll_ns, vote=vote, late=late)
        kernel = wave = WaveMoeTail(batch, tp, twoshot, early, umma, **kw)
        scalars += (vec(3 * CAP * 3584),) if wave.reduce else ()
    elif m8:
        kernel = wave = WaveMoeTail(batch, tp, twoshot, early, umma, late=late)
        scalars += (vec(3 * CAP * 3584),) if wave.reduce else ()
    else:
        kernel = MoeTail(
            tp, batch, mode=mode, trigger=trigger, gate=gate, poll_ns=poll_ns
        )
    options = "" if m8 else "--ptxas-options=--maxrregcount=120"
    with torch.accelerator.device_index(device):
        args = (*tensors, *scalars, make_fake_stream())
        code = cute.compile(kernel, *args, options=options)
        # MoE tail requests 100% shared carveout so the next large PDL kernel can
        # coexist.
        return prepare_shared_carveout(code, device, 100)


@cache
def compile_push(batch, device, tp=8):
    args = (fake((batch, 7168)), fake((tp,), None, Int64), Int32(0), Int32(0))
    with torch.accelerator.device_index(device):
        return cute.compile(
            WaveMoeTail(batch, tp).push, *args, Int32(CAP), make_fake_stream()
        )


class MoeTailWeights:
    """One layer's up-projection shard [7168/TP, 3584], RMS weight and TMA map.
    `dcp`: the deployment's DCP size (the TAIL_GATE / POLL_NS key)."""

    def __init__(
        self, tp, weight, gamma, eps, ordinal, count, batches, twoshots=(), dcp=1
    ):
        self.weight, self.gamma = weight, gamma
        # Three parities rotate over the model's MoE layers; each launch re-arms the
        # previous MoE layer's slots of its batch (wrapping across decode steps, where
        # the first layer re-arms the largest batch the last layer may have published).
        parity, rearm = ordinal % 3, (ordinal - 1) % count % 3
        if parity == rearm:  # a launch must never re-arm the slots it publishes to
            raise ValueError(f"MoE tail parities collide for {count} MoE layers")
        self.parity, rearm = Int32(parity), Int32(rearm)
        self.scalars = (Int32(CAP), Float32(eps), rearm)
        self.wrap = 0 if ordinal else max(batches)
        cols = 512 if tp == 8 else 448
        u16 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16
        dims = (cols // 2, 3584 // (cols // 2), 7168 // tp)
        spec: tuple = (weight.data_ptr(), u16, dims, (cols, 7168), (cols // 2, 2, 32))
        spec += (cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,)
        spec += (cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,)
        self.descriptor = tensor_maps([spec], weight.device)
        self.views = (to_cute(self.descriptor, 64), to_cute(gamma, 16))
        self.codes, device, early = {}, weight.device.index, early_batches(tp)
        umma = umma_batches(tp)
        if any(published(tp, b) and b in umma for b in batches):
            # WaveMoeTail tcgen05: [64 x lines, 64] boxes of the shard, 128B swizzle
            spec = (weight.data_ptr(), u16, (3584, 7168 // tp), (7168,))
            spec += ((64, 128 if tp == 8 else 64),)
            spec += (cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,)
            spec += (cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,)
            self.umma_map = tensor_maps([spec], weight.device)
            self.umma_views = (to_cute(self.umma_map, 64), self.views[1])
        for b in batches:  # WaveMoeTail reads the shard directly, not by TMA
            twoshot = b in twoshots
            mode = "twoshot" if twoshot else "publish" if published(tp, b) else "front"
            views = self.views if mode == "front" else (flat(weight), self.views[1])
            on = mode != "front" and b in umma
            views = self.umma_views if on else views
            gate, sleep = front_gate(tp, dcp, b), front_poll_ns(tp, dcp, b)
            if pub8(tp, b):
                sleep = PUB8_POLL_NS
            kw = dict(early=b in early, umma=on, gate=gate, poll_ns=sleep)
            if on and late(tp, b):
                kw.update(late=True)
            if mode != "front" and tp == 8:  # T8T: only TP8 published tails
                direct8 = direct(tp, b, twoshot)
                sleep8 = published_poll_ns(tp, b)
                vote8 = published_vote(tp, b, twoshot)
                if direct8 or vote8 or sleep8 is not None:
                    kw.update(poll_ns=sleep8, direct=direct8, vote=vote8)
            code = compile_tail(tp, b, device, mode, **kw)
            self.codes[b] = code, views, shares(tp, b, twoshot), twoshot
        shared = any(shares(tp, b, b in twoshots) for b in batches)
        if shared and resident_clusters(device) < 7:
            raise RuntimeError("WaveMoeTail needs seven co-resident 16-CTA clusters")

    def run(self, box, expanded, weights, shared, code=None, batch=None):
        batch = batch or weights.shape[0]
        if code is None:
            code, views, share, twoshot = self.codes[batch]
        else:
            views, share, twoshot = self.views, 0, False
        args, tail = box.args, box.tail
        if twoshot:
            args, tail = box.twoshot_args, box.twoshot_tail
        rows = (flat(expanded), flat(weights), flat(shared), *tail, self.parity)
        rows += (Int32(box.rank), *self.scalars, Int32(self.wrap or batch))
        if share:
            rows += (flat(box.twoshot.gather if twoshot else box.reduced),)
        code(*args, *views, *rows, stream(expanded.device))


class PreparedTail:
    """The layer's MoE tail; placement = (model MoeTailBox, MoE ordinal, MoE layer
    count). At WaveMoeTail batches the experts publish into latent slot `publish[b]`
    (multicast, parity, rank; at `twoshots` the two-shot push box) and the side
    stream pushes the shared-expert output (`push`). `dcp`: the deployment's DCP
    size (MoeTailWeights)."""

    def __init__(self, op, norm, up, batches, placement, dcp=1):
        tp, rank = op.contract.tp_size, op.rank
        local_up = up.narrow(0, rank * (7168 // tp), 7168 // tp)
        self.tail_box, ordinal, count = placement
        eps = op.contract.rms_eps
        m8 = [b for b in batches if published(tp, b)]
        self.twoshots = twoshot_batches(tp) & set(m8)
        args = (tp, local_up, norm, eps, ordinal, count, batches, self.twoshots)
        self.tail_weights = MoeTailWeights(*args, dcp=dcp)
        self.peers, _, mc, _ = self.tail_box.tail
        self.ring = (self.tail_weights.parity, Int32(rank))
        self.pushers = {b: compile_push(b, norm.device.index, tp) for b in m8}
        self.publish = {b: (mc, *self.ring) for b in m8}
        slot = Int32(ordinal % 3 * tp + rank)
        self.publish.update({b: (*self.tail_box.push_box, slot) for b in self.twoshots})

    def push(self, shared):
        if shared.shape[0] in self.pushers:
            args = (to_cute(shared), self.peers, *self.ring, Int32(CAP))
            self.pushers[shared.shape[0]](*args, stream(shared.device))

    def run(self, expanded, weights, shared, out, *, consume):
        self.tail_weights.run(self.tail_box, expanded, weights, shared)
        if consume(out) is not out:
            raise RuntimeError("The tail output must be consumed from its mailbox ring")
        return out
