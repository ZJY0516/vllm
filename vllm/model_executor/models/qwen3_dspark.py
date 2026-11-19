# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DSpark draft model for semi-autoregressive drafting.

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head.

The parallel backbone is a standard Qwen3 decoder stack reused from the
DFlash Qwen3 draft (see qwen3_dflash.py). DSpark adds:
  * `markov_head`: low-rank V x r / r x V transition bias added to the base
    logits, sampled left-to-right by the speculator (the sequential stage).
  * `confidence_head`: per-position acceptance-probability estimate.

DSparkMarkovHead and DSparkConfidenceHead are shared with the DSV4-style DSpark model.
"""

import os
import types
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    nvfp4_gathered_bias,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


class DSparkMarkovHead(nn.Module):
    """Sequential transition-bias head (low-rank V x r, r x V).

    `markov_w1[token]` embeds the previously sampled token (target vocab,
    `vocab_size`); `markov_w2` projects it to a draft-vocab bias
    (`draft_vocab_size`) added to the base draft logits. The two sizes
    coincide for full-vocab drafts.

    Both weights are replicated because the head runs sequentially for every
    draft position. Sharding them would add an all-reduce and a full-vocab
    gather to each position.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        retain_weight_for_gather: bool = False,
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size,
            markov_rank,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "markov_w2"),
            disable_tp=True,
        )
        self.markov_w2._retain_weight_for_gather = retain_weight_for_gather
        self.markov_w2.is_w4a16_nvfp4 = False

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """r-dim Markov embedding of `token_ids` ([B] -> [B, r])."""
        return self.markov_w1(token_ids)

    def bias(
        self,
        markov_embed: torch.Tensor,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """Vocab-size transition bias from a Markov embedding ([B, r] -> [B, V])."""
        return logits_processor(self.markov_w2, markov_embed)

    def apply_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Apply the Markov bias only to selected rows of `logits`.

        The caller initializes `logits` to `-inf` once for all draft
        positions. This method scatters the corrected candidate values into
        that dense buffer so the normal sampler sees the truncated proposal.
        """
        if self.markov_w2._retain_weight_for_gather and self.markov_w2.is_w4a16_nvfp4:
            # Use the fast Triton kernel for W4A16 NVFP4 weights with BF16
            # activations on CUDA. FP16 activations, non-CUDA accelerators, and
            # CPU use the slower dequantize/scatter fallback below.
            if (
                current_platform.is_cuda()
                and markov_embed.is_cuda
                and markov_embed.dtype == torch.bfloat16
                and values.dtype == torch.bfloat16
                and logits.dtype == torch.bfloat16
            ):
                nvfp4_gathered_bias(
                    markov_embed,
                    self.markov_w2._nvfp4_weight_for_gather,
                    self.markov_w2._nvfp4_weight_scale_for_gather,
                    self.markov_w2._nvfp4_weight_global_scale_for_gather,
                    values,
                    index,
                    logits,
                    scale,
                )
                return logits

            flat_index = index.reshape(-1)
            packed_weight = self.markov_w2._nvfp4_weight_for_gather.index_select(
                0, flat_index
            )
            weight_scale = self.markov_w2._nvfp4_weight_scale_for_gather.index_select(
                0, flat_index
            )
            weight = dequantize_to_dtype(
                packed_weight,
                weight_scale,
                self.markov_w2._nvfp4_weight_global_scale_for_gather,
                dtype=markov_embed.dtype,
                block_size=self.markov_w2._nvfp4_group_size_for_gather,
                swizzle=False,
            ).view(*index.shape, -1)
        else:
            weight = self.markov_w2.weight[index]
            if weight.dtype != markov_embed.dtype:
                raise RuntimeError(
                    "Gathered Markov projection requires unquantized weights or "
                    "retained ModelOpt W4A16 NVFP4 weights."
                )
        corrected = values.unsqueeze(-1).clone()
        corrected.baddbmm_(
            weight,
            markov_embed.unsqueeze(-1),
            beta=1.0,
            alpha=scale,
        )
        return logits.scatter_(1, index, corrected.squeeze(-1))


class DSparkConfidenceHead(nn.Module):
    """DSpark acceptance-confidence head."""

    def __init__(
        self,
        input_dim: int,
        prefix: str,
        bias: bool = False,
        with_markov: bool = True,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=bias,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = (
            torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
        ).float()
        return self.proj(x).squeeze(-1)


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone + DSpark Markov / confidence heads."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
            retain_weight_for_gather=(
                getattr(config, "dspark_draft_topk", None) is not None
            ),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", False)
            input_dim = config.hidden_size
            if with_markov:
                input_dim += config.markov_rank
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )
        self._s31_ar_key: int | None = None
        _s31_backbone_init(self)
        _drf_backbone_init(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_drf_norm", False):
            return _drf_forward(self, input_ids, positions, input_embeds)
        if self._s31_ar_key is None:
            return super().forward(input_ids, positions, input_embeds)
        return _s31_forward(self, input_ids, positions, input_embeds)


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = DSparkSpeculatorConfig
    model_cls = Qwen3DSparkModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.draft_model_config = speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        # Only a pruned draft (subset of the target vocab) needs a d2t mapping to
        # scatter ids up. A draft >= the target vocab (e.g. padded to the target's
        # physical embedding size) maps identically, so no d2t is needed.
        if self.config.draft_vocab_size < self.target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None
        _s31_init(self, prefix)

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        _s31_after_loading(self)
        _drf_after_loading(self)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Draft-vocab logits without the d2t scatter: the speculator adds the
        # Markov bias in draft space, then remaps via map_draft_to_target.
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This TP rank's vocab shard of compute_draft_logits, not gathered."""
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        # Map draft-vocab ids to target ids (identity for full-vocab drafts).
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def apply_markov_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.markov_head.apply_bias_gathered(
            markov_embed,
            logits,
            values,
            index,
            self.logits_processor.scale,
        )

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        for name, loaded_weight in weights:
            # t2d is training-only; the draft remaps via d2t at sampling time.
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            model_weights[name] = loaded_weight
            # Sets has_own_embed_tokens / has_own_lm_head so load_dspark_model
            # knows whether to keep these or alias the target's.
            process_eagle_weight(self, name)

        # mask_embedding is an unused placeholder param; DSpark masks via the vocab row.
        # embed_tokens / lm_head are optional; when omitted they are shared from
        # the target by load_dspark_model, so skip the unloaded params here.

        orig_to_new_substr = {"mask_embedding": None}
        if not includes_embed_tokens:
            orig_to_new_substr["embed_tokens"] = None
        if not includes_lm_head:
            orig_to_new_substr["lm_head"] = None
        if not includes_draft_id_mapping:
            orig_to_new_substr["draft_id_to_target_id"] = None
        if self.model.confidence_head is None:
            orig_to_new_substr["confidence_head"] = None
        mapper = WeightsMapper(orig_to_new_substr=orig_to_new_substr)
        loader = AutoWeightsLoader(self)
        loader.load_weights(model_weights.items(), mapper=mapper)


# --------------------------------------------------------------------------------
# Decode fast paths of the DSpark draft (env-gated, default off; TP > 1 only).
#  K31_S31FC=1: the context projection `fc` is column-parallel with its output
#      gathered (each rank streams 1/TP of the weight: the same dot products over
#      the full input width per output column), on the drafter mailbox gather when
#      available, the TP all-gather otherwise.
#  K31_S31MK=1: the sequential Markov sampling runs as the vocabulary-sharded fused
#      launches of `specialized_decode.drafter.markov` (the lm_head shard's logits,
#      the per-step bias + Gumbel select over each rank's vocabulary range, one
#      exchange of the per-rank maxima per step, one gather of the final logits);
#      steps it does not serve keep the loop of the speculator.
# Both need the drafter collectives (`specialized_decode.drafter.comm`), built at
# load on every TP rank in the same order.
# --------------------------------------------------------------------------------


def _s31_on(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0")


def _s31_init(model: "Qwen3DSparkForCausalLM", prefix: str) -> None:
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp = get_tensor_model_parallel_world_size()
    model._s31_fc = False
    model._s31_mk = _s31_on("K31_S31MK") and tp > 1
    backbone = model.model
    fc = getattr(backbone, "fc", None) if backbone.use_aux_hidden_state else None
    if (
        _s31_on("K31_S31FC")
        and tp > 1
        and fc is not None
        and backbone.quant_config is None
        and fc.output_size % tp == 0
    ):
        from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.model import (
            _gathered_linear,
        )

        gathered = _gathered_linear()(
            fc.input_size,
            fc.output_size,
            bias=False,
            gather_output=False,
            return_bias=False,
            params_dtype=fc.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(maybe_prefix(prefix, "model"), "fc"),
        )
        del backbone.fc
        backbone.fc = gathered
        model._s31_fc = True
    if model._s31_mk:
        model.sample_sequential = types.MethodType(_s31_sample_sequential, model)
    if model._s31_fc or model._s31_mk or backbone._s31_dg or (
        backbone._s31_ar_key is not None
    ):
        logger.info(
            "sol31 DSpark draft fast paths: fc column-parallel %s, fused Markov %s, "
            "DraftGemm %s, ArNorm %s (TP %d)",
            model._s31_fc,
            model._s31_mk,
            backbone._s31_dg,
            backbone._s31_ar_key is not None,
            tp,
        )


def _s31_after_loading(model: "Qwen3DSparkForCausalLM") -> None:
    if getattr(model, "_s31_loaded", False):
        return
    model._s31_loaded = True
    backbone = model.model
    if getattr(backbone, "_s31_dg", False):
        count = _s31_install_gemms(backbone)
        logger.info("sol31 DSpark draft linears on DraftGemm: %d", count)
    ar_key = getattr(backbone, "_s31_ar_key", None)
    if not (
        getattr(model, "_s31_fc", False)
        or getattr(model, "_s31_mk", False)
        or ar_key is not None
    ):
        return
    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter import (
        comm as comm_module,
    )

    comm = comm_module.build(model, get_tp_group())
    model._drafter_comm = comm
    if model._s31_fc and comm is not None:
        backbone.fc.comm = comm
    if ar_key is not None:
        _S31_AR[ar_key]["comm"] = comm
    logger.info(
        "sol31 DSpark draft collectives: %s (fc %s, fused Markov %s, ArNorm %s)",
        "mailbox" if comm is not None else "stock",
        model._s31_fc,
        model._s31_mk,
        ar_key is not None,
    )


def _s31_sample_sequential(self, speculator, num_reqs, head_hidden) -> bool:
    """The speculator asks the model for its own Markov sampling first; False
    (nothing done) keeps the speculator's loop."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.markov import (
        sample_sequential as fused,
    )

    return fused(speculator, self, num_reqs, head_hidden)


# K31_S31DG=1: the draft layers' BF16 linears (qkv, o, gate_up, down) run on the
#      drafter decode GEMM (`specialized_decode.drafter.linears.DraftGemm`: warp
#      split-K up to 32 rows, FP32 accumulation, one BF16 rounding; cuBLAS above),
#      called through the custom op `vllm.s31_draft_gemm` so the compiled draft
#      forward keeps it as one opaque call.
# K31_S31AR=1 (1 < TP <= K31_S31AR_MAXTP, default 8): each all-reduce of the draft layers (o_proj, down_proj)
#      runs fused with the RMSNorm that follows it on the drafter mailbox ArNorm
#      (`specialized_decode.drafter.comm.DrafterComm.ar_norm`, BF16 partials summed
#      in FP32 in rank order, one rounding, then the residual add and the RMSNorm),
#      through the custom op `vllm.s31_ar_norm`; rows it does not serve take the
#      TP all-reduce and vLLM's fused add + RMSNorm.

_S31_GEMMS: list = []
_S31_AR: dict = {}
_S31_OPS: set = set()


def _s31_gemm_impl(x: torch.Tensor, key: int) -> torch.Tensor:
    gemm = _S31_GEMMS[key]
    out = torch.empty((x.shape[0], gemm.n), dtype=x.dtype, device=x.device)
    gemm.run(x.contiguous(), out)
    return out


def _s31_gemm_fake(x: torch.Tensor, key: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], _S31_GEMMS[key].n))


def _s31_ar_norm_impl(
    x: torch.Tensor, residual: torch.Tensor, key: int, site: int
) -> tuple[torch.Tensor, torch.Tensor]:
    state = _S31_AR[key]
    norm = state["norms"][site]
    comm = state["comm"]
    out = None
    if comm is not None:
        out = comm.ar_norm(x, residual, norm, site % 2)
    if out is None:
        from vllm import _custom_ops as ops
        from vllm.distributed import tensor_model_parallel_all_reduce

        y = tensor_model_parallel_all_reduce(x.contiguous())
        res = residual.clone()
        ops.fused_add_rms_norm(y, res, norm.weight.data, norm.variance_epsilon)
        out = (y, res)
    return out


def _s31_ar_norm_fake(
    x: torch.Tensor, residual: torch.Tensor, key: int, site: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


def _s31_register_op(name: str) -> None:
    if name in _S31_OPS:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    if name == "s31_draft_gemm":
        direct_register_custom_op(
            "s31_draft_gemm", _s31_gemm_impl, mutates_args=[], fake_impl=_s31_gemm_fake
        )
    elif name == "s31_ar_norm":
        direct_register_custom_op(
            "s31_ar_norm",
            _s31_ar_norm_impl,
            mutates_args=[],
            fake_impl=_s31_ar_norm_fake,
        )
    _S31_OPS.add(name)


def _s31_backbone_init(backbone: "Qwen3DSparkModel") -> None:
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp = get_tensor_model_parallel_world_size()
    backbone._s31_dg = _s31_on("K31_S31DG") and backbone.quant_config is None
    if backbone._s31_dg:
        _s31_register_op("s31_draft_gemm")
    # ArNorm measured slower than the stock all-reduce + RMSNorm at 16 ranks (7 rows)
    # and faster at 8 ranks (8-24 rows): served up to K31_S31AR_MAXTP ranks.
    max_tp = int(os.environ.get("K31_S31AR_MAXTP", "8") or 8)
    if not (_s31_on("K31_S31AR") and 1 < tp <= max_tp):
        return
    if any(
        getattr(layer.self_attn.o_proj, "bias", None) is not None
        or getattr(layer.mlp.down_proj, "bias", None) is not None
        for layer in backbone.layers
    ):
        return
    _s31_register_op("s31_ar_norm")
    norms = []
    for index, layer in enumerate(backbone.layers):
        layer.self_attn.o_proj.reduce_results = False
        layer.mlp.down_proj.reduce_results = False
        if index > 0:
            norms.append(layer.input_layernorm)
        norms.append(layer.post_attention_layernorm)
    norms.append(backbone.norm)
    key = len(_S31_AR)
    _S31_AR[key] = {"norms": norms, "comm": None}
    backbone._s31_ar_key = key


def _s31_forward(backbone, input_ids, positions, input_embeds):
    """`DFlashQwen3Model.forward` with each layer all-reduce fused into the
    RMSNorm after it (`vllm.s31_ar_norm`); same layers, same order."""
    if input_embeds is None:
        input_embeds = backbone.embed_input_ids(input_ids)
    key = backbone._s31_ar_key
    residual = input_embeds
    partial = None
    site = 0
    for index, layer in enumerate(backbone.layers):
        if index == 0:
            hidden = layer.input_layernorm(input_embeds)
        else:
            hidden, residual = torch.ops.vllm.s31_ar_norm(partial, residual, key, site)
            site += 1
        attn = layer.self_attn(positions=positions, hidden_states=hidden)
        hidden, residual = torch.ops.vllm.s31_ar_norm(attn, residual, key, site)
        site += 1
        partial = layer.mlp(hidden)
    hidden, _ = torch.ops.vllm.s31_ar_norm(partial, residual, key, site)
    return hidden


def _s31_gemm_method():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class S31GemmMethod(UnquantizedLinearMethod):
        """An unquantized linear's `apply` on `vllm.s31_draft_gemm` (2-D BF16
        input, no bias); the original method otherwise."""

        def __init__(self, inner, key):  # super().__init__ not needed
            self.inner = inner
            self.key = key

        def __getattr__(self, name):
            return getattr(self.__dict__["inner"], name)

        def apply(self, layer, x, bias=None):
            if bias is not None or x.dim() != 2 or x.dtype != torch.bfloat16:
                return self.inner.apply(layer, x, bias)
            return torch.ops.vllm.s31_draft_gemm(x, self.key)

    return S31GemmMethod


def _s31_install_gemms(backbone) -> int:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.linears import (
        DraftGemm,
    )

    method_cls = _s31_gemm_method()
    count = 0
    for layer in backbone.layers:
        for linear in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        ):
            method = getattr(linear, "quant_method", None)
            weight = getattr(linear, "weight", None)
            if (
                type(method) is not UnquantizedLinearMethod
                or weight is None
                or weight.dtype != torch.bfloat16
                or weight.dim() != 2
                or getattr(linear, "bias", None) is not None
            ):
                continue
            gemm = DraftGemm(weight)
            if not gemm.eligible:
                continue
            key = len(_S31_GEMMS)
            _S31_GEMMS.append(gemm)
            linear.quant_method = method_cls(method, key)
            count += 1
    return count


# --------------------------------------------------------------------------------
# K31_DRF_ATTN=1: each draft layer's attention block -- q RMSNorm,
#      k RMSNorm, RoPE, the K/V cache write of the query rows and the paged
#      sliding-window attention (FA4 prepare + forward + combine) -- runs as one
#      launch (`specialized_decode.drafter.drf_swa.DraftSwaAttention`, Triton, PDL)
#      through the custom op `vllm.drf_swa_attn`. q/k norm, RoPE and the cache
#      write give the stock bits; the attention merges its splits in another order
#      than FA4 (not bit-identical). Steps it does not serve (no attention
#      metadata, non-causal or non-uniform batches, a quantized cache) take the
#      stock path.
# --------------------------------------------------------------------------------

_DRF_ATTN: list = []


def _drf_attn_eligible(attn) -> bool:
    rope = attn.rotary_emb
    return (
        attn.sliding_window is not None
        and attn.causal
        and attn.attention_sink_bias is None
        and attn.v_scale is None
        and attn.head_dim == 64
        and getattr(rope, "head_size", None) == attn.head_dim
        and getattr(rope, "rotary_dim", None) == attn.head_dim
        and getattr(rope, "is_neox_style", False)
        and not getattr(rope, "use_flashinfer", False)
        and attn.q_norm.variance_epsilon == attn.k_norm.variance_epsilon
        and attn.qkv_proj.bias is None
    )


def _drf_backbone_init(backbone: "Qwen3DSparkModel") -> None:
    backbone._drf_attn = False
    if not _s31_on("K31_DRF_ATTN"):
        return
    attns = [layer.self_attn for layer in backbone.layers]
    if not all(_drf_attn_eligible(a) for a in attns):
        logger.info("drf DSpark draft attention: layers not eligible, stock path")
        return
    from vllm.config import get_current_vllm_config

    from vllm.utils.torch_utils import direct_register_custom_op

    if "drf_swa_attn" not in _S31_OPS:
        direct_register_custom_op(
            "drf_swa_attn", _drf_attn_impl, mutates_args=[], fake_impl=_drf_attn_fake
        )
        _S31_OPS.add("drf_swa_attn")
    cfg = get_current_vllm_config()
    max_reqs = int(getattr(cfg.scheduler_config, "max_num_seqs", 1) or 1)
    for attn in attns:
        attn._drf_key = len(_DRF_ATTN)
        _DRF_ATTN.append({"attn": attn, "launcher": None, "max_reqs": max_reqs,
                          "stock": 0, "fused": 0})
        attn.forward = types.MethodType(_drf_attn_forward, attn)
    backbone._drf_attn = True
    logger.info(
        "drf DSpark draft attention: one launch per layer (%d layers, max %d requests)",
        len(attns), max_reqs,
    )


def _drf_after_loading(model: "Qwen3DSparkForCausalLM") -> None:
    backbone = model.model
    if getattr(model, "_drf_loaded", False):
        return
    model._drf_loaded = True
    _drf_install_gemms(model)
    if not getattr(backbone, "_drf_attn", False):
        return
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.drf_swa import (
        DraftSwaAttention,
    )

    shared = None
    for state in _DRF_ATTN:
        attn = state["attn"]
        if state["launcher"] is not None:
            continue
        launcher = DraftSwaAttention(
            attn.num_heads, attn.num_kv_heads, attn.head_dim, attn.sliding_window,
            attn.q_norm.variance_epsilon, attn.scaling, attn.qkv_proj.weight.device,
            max_reqs=state["max_reqs"],
            block_n=int(os.environ.get("K31_DRF_ATTN_BLOCK", "256")),
            tiles=int(os.environ.get("K31_DRF_ATTN_TILES", "1")),
            use_pdl=os.environ.get("K31_DRF_ATTN_PDL", "1") != "0",
            shared=shared,
        )
        shared = shared or launcher
        state["launcher"] = launcher


def _drf_attn_forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    """`DFlashQwen3Attention.forward` with q/k norm, RoPE, the cache write and
    the attention in `vllm.drf_swa_attn`."""
    qkv, _ = self.qkv_proj(hidden_states)
    attn_output = torch.ops.vllm.drf_swa_attn(qkv, positions, self._drf_key)
    output, _ = self.o_proj(attn_output)
    return output


def _drf_attn_stock(attn, qkv, positions):
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q_shape, k_shape = q.shape, k.shape
    q = attn.q_norm(
        q.view(*q_shape[:-1], q_shape[-1] // attn.head_dim, attn.head_dim)
    ).view(q_shape)
    k = attn.k_norm(
        k.view(*k_shape[:-1], k_shape[-1] // attn.head_dim, attn.head_dim)
    ).view(k_shape)
    q, k = attn.rotary_emb(positions, q, k)
    return attn.attn(q, k, v)


def _drf_attn_impl(qkv: torch.Tensor, positions: torch.Tensor, key: int) -> torch.Tensor:
    from vllm.model_executor.layers.attention.attention import get_attention_context

    state = _DRF_ATTN[key]
    attn = state["attn"]
    launcher = state["launcher"]
    md, _, kv_cache, slot_mapping = get_attention_context(attn.attn.layer_name)
    num_tokens = qkv.shape[0]
    n = getattr(md, "max_query_len", 0) if md is not None else 0
    ok = (
        launcher is not None
        and md is not None
        and slot_mapping is not None
        and getattr(md, "causal", None) is True
        and n > 0
        and num_tokens % n == 0
        and qkv.dtype == torch.bfloat16
        and qkv.stride(-1) == 1
        and kv_cache.dtype == torch.bfloat16
        and kv_cache.dim() == 4
        and kv_cache.shape[-1] == 2 * attn.head_dim
        and kv_cache.shape[1] == attn.num_kv_heads
        and getattr(md, "dcp_context_kv_lens", None) is None
    )
    if ok:
        num_reqs = num_tokens // n
        ok = (
            launcher.supports(num_reqs, n)
            and md.query_start_loc.numel() >= num_reqs + 1
            and md.seq_lens.numel() >= num_reqs
            and md.block_table.shape[0] >= num_reqs
            and slot_mapping.numel() >= num_tokens
        )
    if not ok:
        state["stock"] += 1
        if state["stock"] in (1, 1000) and key == 0:
            logger.info("drf DSpark draft attention: stock path (%d calls)", state["stock"])
        return _drf_attn_stock(attn, qkv, positions)
    state["fused"] += 1
    if state["fused"] in (1, 1000, 100000) and key == 0:
        logger.info(
            "drf DSpark draft attention: fused launch (%d calls, %d requests x %d rows, "
            "page %d)", state["fused"], num_reqs, n, kv_cache.shape[2],
        )
    out = torch.empty(num_tokens, attn.q_size, dtype=qkv.dtype, device=qkv.device)
    cos_sin = attn.rotary_emb._match_cos_sin_cache_dtype(qkv)
    launcher(
        qkv, positions, attn.q_norm.weight, attn.k_norm.weight, cos_sin, kv_cache,
        md.block_table, md.seq_lens, md.query_start_loc, slot_mapping, out,
        num_reqs, n,
    )
    return out


def _drf_attn_fake(qkv: torch.Tensor, positions: torch.Tensor, key: int) -> torch.Tensor:
    attn = _DRF_ATTN[key]["attn"]
    return qkv.new_empty((qkv.shape[0], attn.q_size))


# --------------------------------------------------------------------------------
# K31_DRF_ACT=1: each draft layer's gate_up projection runs with the
#      SiLU-and-mul in its epilogue (`specialized_decode.drafter.drf_gemm.DrfGemm`,
#      gate/up weight rows interleaved per 16-row output tile in place at load,
#      silu from a table made by vLLM's own silu_and_mul): bit-identical to the
#      stock GEMV + act_and_mul, one launch fewer per layer.
# K31_DRF_NORM=1 (TP > 1 without K31_S31AR): the residual add and the
#      RMSNorms of the draft layers move into the projections around them: tp rank 0
#      adds the residual to its o_proj / down_proj partial (the all-reduce returns
#      the new residual stream r), and the next projection (qkv, gate_up) reads r
#      with the RMSNorm weight folded into its weight columns and scales its
#      accumulator by rsqrt(mean(r^2) + eps) (`DrfGemm(norm=True)`). Removes the
#      ten fused_add_rms_norm launches and the first rms_norm; the final norm stays.
#      Not bit-identical (roundings move); the context K/V path keeps its own
#      unfolded copy of the K/V weights.
# Rows past 32 take a torch fallback of the same math.
# --------------------------------------------------------------------------------

_DRF_GEMMS: list = []


def _drf_tactic(n, k, tile):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.decode import (
        warp_tactic,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.warp_splitk import (
        WarpSplitKTactic,
    )

    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    def tactic(m):
        t = warp_tactic(m, n, k, sms)
        return WarpSplitKTactic(tile, t.token_tile, t.k_tile, t.stages, t.b_loader_warps)

    return tactic


def _drf_register_gemm(weight, *, norm=False, resid=False, act=False, eps=1e-5,
                       tactic_fn=None, tile=16) -> int:
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.drf_gemm import DrfGemm

    engine = None
    try:
        engine = DrfGemm(weight, norm=norm, resid=resid, act=act, eps=eps,
                         tactic_fn=tactic_fn)
    except Exception as exc:  # pragma: no cover - capability fallback
        logger.warning("drf DSpark draft GEMM %s on torch fallback: %s",
                       tuple(weight.shape), exc)
    n = weight.shape[0]
    _DRF_GEMMS.append({
        "engine": engine, "weight": weight, "norm": norm, "resid": resid,
        "act": act, "eps": float(eps), "n_out": n // 2 if act else n, "tile": tile,
        "kernel": 0, "fallback": 0,
    })
    return len(_DRF_GEMMS) - 1


def _drf_gemm_fallback(st, x, residual):
    w = st["weight"]
    y = torch.mm(x, w.t())
    if st["norm"]:
        rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + st["eps"])
        y = (y.float() * rstd).to(x.dtype)
    if st["resid"] and residual is not None:
        y = (y.float() + residual.float()).to(x.dtype)
    if st["act"]:
        m = y.shape[0]
        half = st["tile"] // 2
        y4 = y.view(m, -1, 2, half)
        gate = y4[:, :, 0, :].reshape(m, -1)
        up = y4[:, :, 1, :].reshape(m, -1)
        out = torch.empty(m, st["n_out"], dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, torch.cat([gate, up], dim=-1))
        return out
    return y


def _drf_gemm_impl(
    x: torch.Tensor, key: int, residual: torch.Tensor | None
) -> torch.Tensor:
    st = _DRF_GEMMS[key]
    engine = st["engine"]
    m = x.shape[0]
    capturing = torch.cuda.is_current_stream_capturing()
    if (
        engine is not None
        and 1 <= m <= 32
        and (m in engine.codes or not capturing)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 32 == 0
        and (not st["resid"] or (residual is not None and residual.is_contiguous()
                                 and residual.data_ptr() % 32 == 0))
    ):
        out = torch.empty(m, st["n_out"], dtype=x.dtype, device=x.device)
        engine.run(x, out, residual if st["resid"] else None)
        st["kernel"] += 1
        return out
    st["fallback"] += 1
    if st["fallback"] in (1, 1000):
        logger.info("drf DSpark draft GEMM %d: torch fallback (%d rows, %d calls)",
                    key, m, st["fallback"])
    return _drf_gemm_fallback(st, x, residual)


def _drf_gemm_fake(
    x: torch.Tensor, key: int, residual: torch.Tensor | None
) -> torch.Tensor:
    return x.new_empty((x.shape[0], _DRF_GEMMS[key]["n_out"]))


def _drf_register_gemm_op() -> None:
    if "drf_gemm" in _S31_OPS:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        "drf_gemm", _drf_gemm_impl, mutates_args=[], fake_impl=_drf_gemm_fake
    )
    _S31_OPS.add("drf_gemm")


def _drf_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Qwen2MLP.forward with the SiLU-and-mul in the gate_up epilogue."""
    h = torch.ops.vllm.drf_gemm(x, self._drf_gu_key, None)
    out, _ = self.down_proj(h)
    return out


def _drf_install_gemms(model: "Qwen3DSparkForCausalLM") -> None:
    backbone = model.model
    want_norm = _s31_on("K31_DRF_NORM")
    want_act = _s31_on("K31_DRF_ACT") or want_norm
    backbone._drf_norm = False
    if not (want_norm or want_act):
        return
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    def plain(linear):
        method = getattr(linear, "quant_method", None)
        inner = getattr(method, "inner", method)
        w = getattr(linear, "weight", None)
        return (
            isinstance(inner, UnquantizedLinearMethod)
            and w is not None and w.dtype == torch.bfloat16 and w.dim() == 2
            and getattr(linear, "bias", None) is None
        )

    layers = list(backbone.layers)
    tp = get_tensor_model_parallel_world_size()
    tile = int(os.environ.get("K31_DRF_ACT_TILE", "16" if tp >= 16 else "32"))
    rank0 = get_tensor_model_parallel_rank() == 0
    ok_act = all(
        plain(layer.mlp.gate_up_proj) and plain(layer.mlp.down_proj)
        and type(layer.mlp.act_fn).__name__ == "SiluAndMul"
        and layer.mlp.gate_up_proj.weight.shape[0] % (2 * tile) == 0
        for layer in layers
    )
    ok_norm = (
        want_norm and ok_act and tp > 1 and backbone._s31_ar_key is None
        and getattr(backbone, "_fused_kv_weight", None) is not None
        and all(plain(layer.self_attn.qkv_proj) and plain(layer.self_attn.o_proj)
                and layer.input_layernorm.weight.shape[0] == 7168 for layer in layers)
    )
    if not ok_act:
        logger.info("drf DSpark draft GEMMs: layers not eligible, stock path")
        return
    _drf_register_gemm_op()
    for layer in layers:
        mlp = layer.mlp
        w = mlp.gate_up_proj.weight
        eps = layer.post_attention_layernorm.variance_epsilon
        with torch.no_grad():
            if ok_norm:
                w.mul_(layer.post_attention_layernorm.weight.to(w.dtype)[None, :])
            from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.drf_gemm import (
                permute_gate_up,
            )

            w.copy_(permute_gate_up(w, tile))
        mlp._drf_gu_key = _drf_register_gemm(
            w, norm=ok_norm, act=True, eps=eps, tactic_fn=_drf_tactic(*w.shape, tile),
            tile=tile,
        )
        mlp.forward = types.MethodType(_drf_mlp_forward, mlp)
        if ok_norm:
            attn = layer.self_attn
            wq = attn.qkv_proj.weight
            with torch.no_grad():
                wq.mul_(layer.input_layernorm.weight.to(wq.dtype)[None, :])
            attn._drf_qkv_key = _drf_register_gemm(
                wq, norm=True, eps=layer.input_layernorm.variance_epsilon,
                tactic_fn=_drf_tactic(*wq.shape, 16),
            )
            attn._drf_o_key = _drf_register_gemm(attn.o_proj.weight, resid=rank0)
            mlp._drf_down_key = _drf_register_gemm(mlp.down_proj.weight, resid=rank0)
    backbone._drf_norm = ok_norm
    logger.info(
        "drf DSpark draft GEMMs: SiLU-mul in gate_up %s (tile %d), RMSNorm folded %s "
        "(TP %d, rank0 residual %s)", True, tile, ok_norm, tp, rank0,
    )


def _drf_forward(backbone, input_ids, positions, input_embeds):
    """`DFlashQwen3Model.forward` with the residual adds and RMSNorms inside the
    projections (K31_DRF_NORM); same layers, same order, same all-reduces."""
    from vllm.distributed import tensor_model_parallel_all_reduce

    if input_embeds is None:
        input_embeds = backbone.embed_input_ids(input_ids)
    r = input_embeds.contiguous()
    use_attn = getattr(backbone, "_drf_attn", False)
    for layer in backbone.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        qkv = torch.ops.vllm.drf_gemm(r, attn._drf_qkv_key, None)
        if use_attn:
            a = torch.ops.vllm.drf_swa_attn(qkv, positions, attn._drf_key)
        else:
            a = _drf_attn_stock(attn, qkv, positions)
        r = tensor_model_parallel_all_reduce(
            torch.ops.vllm.drf_gemm(a.contiguous(), attn._drf_o_key, r)
        )
        h = torch.ops.vllm.drf_gemm(r, mlp._drf_gu_key, None)
        r = tensor_model_parallel_all_reduce(
            torch.ops.vllm.drf_gemm(h, mlp._drf_down_key, r)
        )
    return backbone.norm(r)
