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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
