# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3's model runner state: the hybrid (KDA + MLA) state, plus the lifecycle of
the specialized decode when `decode_backend=latency_specialized` selects it."""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

logger = init_logger(__name__)


# Fused post-sample state update (K31_HP_PS=0 turns it off).
# After sampling, the hybrid state runs, per request, `_scatter_num_accepted_kernel`
# and then, for each KDA metadata group with the deferred RecoverSSM, the commit
# plan (`_prepare_commit_plan_kernel`), the deferral (`_defer_commit_kernel`) and
# the align postprocess (`_postprocess_recoverssm_align_kernel`): 1 + 3 * groups
# launches. `_post_sample_kernel` runs that sequence as one launch, one program per
# request, with the same per-request statements in the same order (each launch is
# request-parallel and requests touch disjoint slots), so every buffer ends with the
# same bits.
def _post_sample_fused_enabled() -> bool:
    import os

    try:
        return int(os.environ.get("K31_HP_PS", "1") or 0) > 0
    except ValueError:
        return False


@triton.jit
def _post_sample_group(
    b,
    num_sampled,
    req_state_idx,
    idx_ok,
    num_sampled_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    pending_ptr,
    st_ptr,
    st_stride,
    qsl_ptr,
    qsl_stride,
    bt_ptr,
    bt_s0,
    bt_s1,
    bt_width,
    nc_ptr,
    nc_stride,
    mbs,
    cl_ptr,
    fs_ptr,
    bs_ptr,
    br_ptr,
    rr_ptr,
    null_block_id,
    ns_stride,
    SPEC_QUERY_LEN: tl.constexpr,
    ALIGN_BS: tl.constexpr,
    ALIGN_W: tl.constexpr,
):
    # == _prepare_commit_plan_kernel (no request indices, align mode), program b
    source_state_idx = tl.load(st_ptr + b * st_stride).to(tl.int64)
    num_accepted = tl.load(num_sampled_ptr + b * ns_stride).to(tl.int32)
    bos = tl.load(qsl_ptr + b * qsl_stride).to(tl.int64)
    eos = tl.load(qsl_ptr + (b + 1) * qsl_stride).to(tl.int64)
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)
    num_computed = tl.load(nc_ptr + b * nc_stride).to(tl.int32)
    final_num_computed = num_computed + commit_len
    final_state_col = tl.minimum(
        tl.maximum(final_num_computed - 1, 0) // mbs,
        bt_width - 1,
    )
    final_state_idx = tl.load(bt_ptr + b * bt_s0 + final_state_col * bt_s1).to(tl.int64)
    next_boundary = (num_computed // mbs + 1) * mbs
    crosses_boundary = final_num_computed >= next_boundary
    boundary_recovery_len = next_boundary - num_computed
    boundary_state_idx = tl.load(
        bt_ptr + b * bt_s0 + (next_boundary // mbs - 1) * bt_s1,
        mask=crosses_boundary,
        other=null_block_id,
    ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    planned = tl.where(valid, commit_len, 0)
    tl.store(cl_ptr + b, planned)
    tl.store(fs_ptr + b, tl.where(valid, final_state_idx, null_block_id))
    tl.store(bs_ptr + b, tl.where(valid, boundary_state_idx, null_block_id))
    tl.store(br_ptr + b, tl.where(valid, boundary_recovery_len, 0))
    # == _defer_commit_kernel (no eager rows), program b: reads back the plan's
    # commit length (`planned`, the value just stored)
    record_row = tl.load(rr_ptr + b)
    defer_src = tl.load(st_ptr + b * st_stride)
    if (record_row >= 0) & (defer_src > null_block_id):
        tl.store(pending_ptr + record_row, planned)
        tl.store(cl_ptr + b, 0)
    # == _postprocess_recoverssm_align_kernel (no request indices), program b
    if idx_ok:
        nc_align = tl.load(nc_ptr + b)
        tl.store(
            state_idx_ptr + req_state_idx,
            tl.minimum(
                tl.maximum(nc_align + num_sampled - 1, 0) // ALIGN_BS,
                ALIGN_W - 1,
            ),
        )
        tl.store(num_accepted_ptr + req_state_idx, 1)


@triton.jit
def _post_sample_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    ns_stride,
    num_accepted_ptr,
    state_idx_ptr,
    pending_ptr,
    null_block_id,
    st0, sts0, q0, qs0, bt0, b00, b10, w0, nc0, ncs0, m0, cl0, fs0, bs0, br0, rr0,
    st1, sts1, q1, qs1, bt1, b01, b11, w1, nc1, ncs1, m1, cl1, fs1, bs1, br1, rr1,
    st2, sts2, q2, qs2, bt2, b02, b12, w2, nc2, ncs2, m2, cl2, fs2, bs2, br2, rr2,
    st3, sts3, q3, qs3, bt3, b03, b13, w3, nc3, ncs3, m3, cl3, fs3, bs3, br3, rr3,
    NG: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    ABS0: tl.constexpr, AW0: tl.constexpr,
    ABS1: tl.constexpr, AW1: tl.constexpr,
    ABS2: tl.constexpr, AW2: tl.constexpr,
    ABS3: tl.constexpr, AW3: tl.constexpr,
):
    b = tl.program_id(0)
    # == _scatter_num_accepted_kernel, program b
    req_state_idx = tl.load(idx_mapping_ptr + b)
    idx_ok = req_state_idx >= 0
    num_sampled = tl.load(num_sampled_ptr + b)
    if idx_ok:
        tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))
    _post_sample_group(
        b, num_sampled, req_state_idx, idx_ok, num_sampled_ptr, state_idx_ptr,
        num_accepted_ptr, pending_ptr, st0, sts0, q0, qs0, bt0, b00, b10, w0,
        nc0, ncs0, m0, cl0, fs0, bs0, br0, rr0, null_block_id, ns_stride,
        SPEC_QUERY_LEN, ABS0, AW0,
    )  # fmt: skip
    if NG > 1:
        _post_sample_group(
            b, num_sampled, req_state_idx, idx_ok, num_sampled_ptr, state_idx_ptr,
            num_accepted_ptr, pending_ptr, st1, sts1, q1, qs1, bt1, b01, b11, w1,
            nc1, ncs1, m1, cl1, fs1, bs1, br1, rr1, null_block_id, ns_stride,
            SPEC_QUERY_LEN, ABS1, AW1,
        )  # fmt: skip
    if NG > 2:
        _post_sample_group(
            b, num_sampled, req_state_idx, idx_ok, num_sampled_ptr, state_idx_ptr,
            num_accepted_ptr, pending_ptr, st2, sts2, q2, qs2, bt2, b02, b12, w2,
            nc2, ncs2, m2, cl2, fs2, bs2, br2, rr2, null_block_id, ns_stride,
            SPEC_QUERY_LEN, ABS2, AW2,
        )  # fmt: skip
    if NG > 3:
        _post_sample_group(
            b, num_sampled, req_state_idx, idx_ok, num_sampled_ptr, state_idx_ptr,
            num_accepted_ptr, pending_ptr, st3, sts3, q3, qs3, bt3, b03, b13, w3,
            nc3, ncs3, m3, cl3, fs3, bs3, br3, rr3, null_block_id, ns_stride,
            SPEC_QUERY_LEN, ABS3, AW3,
        )  # fmt: skip


def _post_sample_groups(step, num_reqs: int):
    """The fused path's per-group operands, or None when a group needs the stock
    sequence (eager commit rows, request indices, the Triton backend, ...)."""
    from .specialized_decode.recoverssm.context import FP32_DEFERRED
    from .specialized_decode.recoverssm.state import _DeferredStepCommit

    if not step or len(step) > 4:
        return None
    groups = []
    spec_len = None
    for md in step:
        commit = getattr(md, "recoverssm_commit", None)
        step_ctx = getattr(md, "recoverssm_context", None)
        if commit is None or commit.align is None or commit.request_indices is not None:
            return None
        if not isinstance(step_ctx, _DeferredStepCommit):
            return None
        ctx, deferred = step_ctx.context, step_ctx.deferred
        if ctx.backend != FP32_DEFERRED or deferred.eager is not None:
            return None
        n = md.num_spec_decodes
        align = commit.align
        st = commit.state_indices[:n, 0]
        qsl = commit.query_start_loc[: n + 1]
        bt, nc = align.block_table, align.num_computed_tokens
        if (
            n != num_reqs
            or n > ctx.commit_lens.shape[0]
            or qsl.shape[0] != n + 1
            or bt.ndim != 2
            or nc.stride(0) != 1
            or align.block_size < ctx.spec_query_len
            or (spec_len is not None and spec_len != ctx.spec_query_len)
        ):
            return None
        spec_len = ctx.spec_query_len
        groups.append(
            (
                st,
                qsl,
                bt,
                nc,
                align.block_size,
                ctx,
                deferred.record_rows,
                deferred.pending,
            )
        )
    return groups, spec_len


def _post_sample_launch(
    idx_mapping, num_sampled, num_accepted, state_idx, groups, spec_len
):
    pending = groups[0][7]
    if any(g[7] is not pending for g in groups):
        return False
    args = []
    consts = {}
    for k in range(4):
        st, qsl, bt, nc, mbs, ctx, rr, _ = groups[min(k, len(groups) - 1)]
        args += [
            st, st.stride(0), qsl, qsl.stride(0), bt, bt.stride(0), bt.stride(1),
            bt.shape[1], nc, nc.stride(0), mbs, ctx.commit_lens,
            ctx.final_state_indices, ctx.boundary_state_indices,
            ctx.boundary_recovery_lens, rr,
        ]
        consts[f"ABS{k}"] = mbs
        consts[f"AW{k}"] = bt.shape[1]
    _post_sample_kernel[(idx_mapping.shape[0],)](
        idx_mapping,
        num_sampled,
        num_sampled.stride(0),
        num_accepted,
        state_idx,
        pending,
        NULL_BLOCK_ID,
        *args,
        NG=len(groups),
        SPEC_QUERY_LEN=spec_len,
        num_warps=1,
        **consts,
    )
    return True


class KimiK3ModelState(MambaHybridModelState):
    """Builds the specialized decode once the weights are loaded and the aux layers
    set, and prepares it for every bound KV cache (the profiling cache, then the
    real one)."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        # Fused post-sample state update: enabled flag read at the first step,
        # counters for the log.
        self._post_sample_fused: bool | None = None
        self._post_sample_steps = self._post_sample_fallbacks = 0
        if model.specialized_decode_config is not None:
            model.build_specialized_decode()
            model.specialized_decode.bind_model_state(self)

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.prepare_kv()

    # -- Fused post-sample state update (K31_HP_PS) ---------------------------------
    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
        enabled = self._post_sample_fused
        if enabled is None:
            enabled = self._post_sample_fused = _post_sample_fused_enabled()
        if enabled and self._fused_post_sample(
            idx_mapping, num_sampled, num_computed_tokens
        ):
            return
        super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)

    def _fused_post_sample(self, idx_mapping, num_sampled, num_computed_tokens) -> bool:
        """The stock post-sample sequence of `MambaHybridModelState.postprocess_state`
        with its scatter, RecoverSSM commits and align postprocesses in one launch;
        False: run the stock sequence."""
        recoverssm = self.recoverssm
        step = getattr(recoverssm, "_step", None)
        if (
            isinstance(num_sampled, int)
            or step is None
            or not self._align_mode
            or self._mamba_state_idx_gpu is None
        ):
            return False
        num_reqs = idx_mapping.shape[0]
        if num_reqs == 0:
            return False
        found = _post_sample_groups(step, num_reqs)
        if found is None:
            self._post_sample_fallbacks += 1
            return False
        groups, spec_len = found
        if not _post_sample_launch(
            idx_mapping,
            num_sampled,
            self.num_accepted_tokens_gpu,
            self._mamba_state_idx_gpu,
            groups,
            spec_len,
        ):
            self._post_sample_fallbacks += 1
            return False
        recoverssm._step = None
        self._post_sample_steps += 1
        if self._post_sample_steps in (1, 1000):
            logger.info(
                "Kimi K3: %d fused post-sample steps (%d groups), %d stock fallbacks",
                self._post_sample_steps,
                len(groups),
                self._post_sample_fallbacks,
            )
        # The rest of MambaHybridModelState.postprocess_state, unchanged.
        if num_computed_tokens is not None and self._mamba_ctx is not None:
            self._mamba_ctx.run_fused_postprocess_align(
                num_reqs,
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )
        self._publish_replayssm_trackers(idx_mapping, num_computed_tokens)
        return True
