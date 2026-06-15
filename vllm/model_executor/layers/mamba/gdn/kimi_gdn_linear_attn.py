# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from einops import rearrange
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from ...fla.ops.kda import (
    FusedRMSNormGated,
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
)
from ...linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from ..mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from ..ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

logger = init_logger(__name__)


def kda_attention(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(
        q_proj_states=q_proj_states,
        k_proj_states=k_proj_states,
        v_proj_states=v_proj_states,
        g1=g1,
        beta=beta,
        core_attn_out=core_attn_out,
    )


def kda_attention_fake(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="kda_attention",
    op_func=kda_attention,
    mutates_args=["core_attn_out"],
    fake_impl=kda_attention_fake,
)


@PluggableLayer.register("kimi_gated_delta_net_attention")
class KimiGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size, self.num_heads, self.head_dim, conv_kernel_size=self.conv_size
        )

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = kda_config["short_conv_kernel_size"]

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_a_proj",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.b_proj",
        )

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]

        beta = self.b_proj(hidden_states)[0].float().sigmoid()
        g1 = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q,
            k,
            v,
            g1,
            beta,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            #     # V1 profile run
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        constant_caches = self.kv_cache

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_state_q, conv_state_k, conv_state_v = conv_state.chunk(3, dim=-2)

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )
        if attn_metadata_narrowed.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            q, k, v = self._causal_conv1d_prefill_with_optional_checkpoint(
                q_proj_states=q_proj_states,
                k_proj_states=k_proj_states,
                v_proj_states=v_proj_states,
                q_conv_weights=q_conv_weights,
                k_conv_weights=k_conv_weights,
                v_conv_weights=v_conv_weights,
                conv_state_q=conv_state_q,
                conv_state_k=conv_state_k,
                conv_state_v=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                attn_metadata=attn_metadata_narrowed,
                num_actual_tokens=num_actual_tokens,
            )
        else:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_actual_tokens
            ]
            q = causal_conv1d_update(
                q_proj_states,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim), (q, k, v)
        )

        if attn_metadata_narrowed.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            zero_idx = non_spec_state_indices_tensor[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
            core_attn_out_non_spec, last_recurrent_state = (
                self._chunk_kda_prefill_with_optional_internal_split(
                    q=q,
                    k=k,
                    v=v,
                    raw_g=g1,
                    beta=beta,
                    initial_state=initial_state,
                    cu_seqlens=non_spec_query_start_loc,
                    attn_metadata=attn_metadata_narrowed,
                    recurrent_state=recurrent_state,
                )
            )
            # Init cache
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        else:
            assert non_spec_query_start_loc is not None
            g1 = fused_kda_gate(
                rearrange(g1, "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
            ).unsqueeze(0)
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata_narrowed.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
            0, :num_actual_tokens
        ]

    @staticmethod
    def _make_cu(lengths: list[int], device: torch.device) -> torch.Tensor:
        values = [0]
        for length in lengths:
            values.append(values[-1] + length)
        return torch.tensor(values, dtype=torch.int32, device=device)

    @staticmethod
    def _select_segments_4d(
        tensors: tuple[torch.Tensor, ...],
        segments: list[tuple[int, int, int, bool]],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.cat([tensor[:, start:end] for start, end, _, _ in segments], dim=1)
            for tensor in tensors
        )

    @staticmethod
    def _select_segments_2d(
        tensor: torch.Tensor,
        segments: list[tuple[int, int, int, bool]],
    ) -> torch.Tensor:
        return torch.cat([tensor[start:end] for start, end, _, _ in segments], dim=0)

    @staticmethod
    def _get_non_spec_checkpoint_splits(
        attn_metadata: GDNAttentionMetadata,
    ) -> list[tuple[int, int]]:
        query_start_loc_cpu = attn_metadata.non_spec_query_start_loc_cpu
        checkpoint_offsets_cpu = attn_metadata.non_spec_checkpoint_offsets_cpu
        checkpoint_state_indices_cpu = (
            attn_metadata.non_spec_checkpoint_state_indices_cpu
        )
        if (
            query_start_loc_cpu is None
            or checkpoint_offsets_cpu is None
            or checkpoint_state_indices_cpu is None
        ):
            return []

        num_rows = min(
            query_start_loc_cpu.numel() - 1,
            checkpoint_offsets_cpu.numel(),
            checkpoint_state_indices_cpu.numel(),
        )
        splits: list[tuple[int, int]] = []
        for row_idx in range(num_rows):
            checkpoint_state_idx = int(checkpoint_state_indices_cpu[row_idx].item())
            offset = int(checkpoint_offsets_cpu[row_idx].item())
            row_len = int(
                (query_start_loc_cpu[row_idx + 1] - query_start_loc_cpu[row_idx]).item()
            )
            if checkpoint_state_idx >= 0 and 0 < offset < row_len:
                splits.append((row_idx, offset))
        return splits

    def _chunk_kda_prefill_with_optional_internal_split(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        splits = self._get_non_spec_checkpoint_splits(attn_metadata)
        if not splits:
            return chunk_kda_with_fused_gate(
                q=q,
                k=k,
                v=v,
                raw_g=raw_g,
                beta=beta,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        device = q.device
        phase1_segments: list[tuple[int, int, int, bool]] = []
        phase2_segments: list[tuple[int, int, int, bool]] = []
        assert attn_metadata.non_spec_query_start_loc_cpu is not None
        query_start_loc_cpu = attn_metadata.non_spec_query_start_loc_cpu
        num_rows = min(initial_state.shape[0], query_start_loc_cpu.numel() - 1)
        splits = [(row_idx, offset) for row_idx, offset in splits if row_idx < num_rows]
        if not splits:
            return chunk_kda_with_fused_gate(
                q=q,
                k=k,
                v=v,
                raw_g=raw_g,
                beta=beta,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        split_by_row = {row_idx: offset for row_idx, offset in splits}
        for row_idx in range(num_rows):
            start = int(query_start_loc_cpu[row_idx].item())
            end = int(query_start_loc_cpu[row_idx + 1].item())
            if start >= end:
                continue
            offset = split_by_row.get(row_idx)
            if offset is None:
                phase1_segments.append((start, end, row_idx, False))
            else:
                split_abs = start + offset
                phase1_segments.append((start, split_abs, row_idx, True))
                phase2_segments.append((split_abs, end, row_idx, False))

        phase1_q, phase1_k, phase1_v = self._select_segments_4d(
            (q, k, v), phase1_segments
        )
        phase1_raw_g, phase1_beta = self._select_segments_4d(
            (raw_g, beta), phase1_segments
        )
        phase1_initial_state = initial_state[
            torch.tensor(
                [row_idx for _, _, row_idx, _ in phase1_segments],
                dtype=torch.long,
                device=device,
            )
        ]
        phase1_out, phase1_final_state = chunk_kda_with_fused_gate(
            q=phase1_q,
            k=phase1_k,
            v=phase1_v,
            raw_g=phase1_raw_g,
            beta=phase1_beta,
            A_log=self.A_log,
            g_bias=self.dt_bias,
            initial_state=phase1_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=self._make_cu(
                [end - start for start, end, _, _ in phase1_segments], device
            ),
        )

        output = torch.empty_like(q)
        last_recurrent_state = initial_state.clone()
        checkpoint_states: list[torch.Tensor] = []
        checkpoint_rows: list[int] = []
        phase1_offset = 0
        for seg_idx, (start, end, row_idx, is_checkpoint) in enumerate(phase1_segments):
            length = end - start
            output[:, start:end] = phase1_out[:, phase1_offset : phase1_offset + length]
            phase1_offset += length
            if is_checkpoint:
                checkpoint_states.append(phase1_final_state[seg_idx])
                checkpoint_rows.append(row_idx)
            else:
                last_recurrent_state[row_idx] = phase1_final_state[seg_idx]

        if not phase2_segments:
            return output, last_recurrent_state

        assert attn_metadata.non_spec_checkpoint_state_indices is not None
        checkpoint_row_tensor = torch.tensor(
            checkpoint_rows, dtype=torch.long, device=device
        )
        checkpoint_state_indices = attn_metadata.non_spec_checkpoint_state_indices[
            checkpoint_row_tensor
        ].long()
        checkpoint_state_tensor = torch.stack(checkpoint_states, dim=0)
        recurrent_state[checkpoint_state_indices] = checkpoint_state_tensor.to(
            recurrent_state.dtype
        )

        phase2_q, phase2_k, phase2_v = self._select_segments_4d(
            (q, k, v), phase2_segments
        )
        phase2_raw_g, phase2_beta = self._select_segments_4d(
            (raw_g, beta), phase2_segments
        )
        second_out, phase2_final_state = chunk_kda_with_fused_gate(
            q=phase2_q,
            k=phase2_k,
            v=phase2_v,
            raw_g=phase2_raw_g,
            beta=phase2_beta,
            A_log=self.A_log,
            g_bias=self.dt_bias,
            initial_state=checkpoint_state_tensor,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=self._make_cu(
                [end - start for start, end, _, _ in phase2_segments], device
            ),
        )
        phase2_offset = 0
        for seg_idx, (start, end, row_idx, _) in enumerate(phase2_segments):
            length = end - start
            output[:, start:end] = second_out[:, phase2_offset : phase2_offset + length]
            phase2_offset += length
            last_recurrent_state[row_idx] = phase2_final_state[seg_idx]
        return output, last_recurrent_state

    def _causal_conv1d_prefill_with_optional_checkpoint(
        self,
        *,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        q_conv_weights: torch.Tensor,
        k_conv_weights: torch.Tensor,
        v_conv_weights: torch.Tensor,
        conv_state_q: torch.Tensor,
        conv_state_k: torch.Tensor,
        conv_state_v: torch.Tensor,
        has_initial_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        attn_metadata: GDNAttentionMetadata,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        splits = self._get_non_spec_checkpoint_splits(attn_metadata)
        if not splits or query_start_loc is None:
            q_proj_states = q_proj_states.transpose(0, 1)
            k_proj_states = k_proj_states.transpose(0, 1)
            v_proj_states = v_proj_states.transpose(0, 1)
            q = causal_conv1d_fn(
                q_proj_states,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            return q, k, v

        device = q_proj_states.device
        assert attn_metadata.non_spec_query_start_loc_cpu is not None
        query_start_loc_cpu = attn_metadata.non_spec_query_start_loc_cpu
        num_rows = min(cache_indices.numel(), query_start_loc_cpu.numel() - 1)
        splits = [(row_idx, offset) for row_idx, offset in splits if row_idx < num_rows]
        if not splits:
            q_proj_states = q_proj_states.transpose(0, 1)
            k_proj_states = k_proj_states.transpose(0, 1)
            v_proj_states = v_proj_states.transpose(0, 1)
            q = causal_conv1d_fn(
                q_proj_states,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
            return q, k, v
        split_by_row = {row_idx: offset for row_idx, offset in splits}
        phase1_segments: list[tuple[int, int, int, bool]] = []
        phase2_segments: list[tuple[int, int, int, bool]] = []
        for row_idx in range(num_rows):
            start = int(query_start_loc_cpu[row_idx].item())
            end = int(query_start_loc_cpu[row_idx + 1].item())
            if start >= end:
                continue
            offset = split_by_row.get(row_idx)
            if offset is None:
                phase1_segments.append((start, end, row_idx, False))
            else:
                split_abs = start + offset
                phase1_segments.append((start, split_abs, row_idx, True))
                phase2_segments.append((split_abs, end, row_idx, False))

        phase1_req_indices = torch.tensor(
            [row_idx for _, _, row_idx, _ in phase1_segments],
            dtype=torch.long,
            device=device,
        )
        checkpoint_rows = [row_idx for row_idx, _ in splits]
        checkpoint_row_tensor = torch.tensor(
            checkpoint_rows,
            dtype=torch.long,
            device=device,
        )
        assert attn_metadata.non_spec_checkpoint_state_indices is not None
        checkpoint_state_indices = attn_metadata.non_spec_checkpoint_state_indices[
            checkpoint_row_tensor
        ].long()
        source_state_indices = cache_indices[checkpoint_row_tensor].long()

        def build_cache_indices(
            segments: list[tuple[int, int, int, bool]],
        ) -> torch.Tensor:
            pieces = []
            for _, _, row_idx, is_checkpoint in segments:
                if is_checkpoint:
                    assert attn_metadata.non_spec_checkpoint_state_indices is not None
                    pieces.append(
                        attn_metadata.non_spec_checkpoint_state_indices[
                            row_idx : row_idx + 1
                        ]
                    )
                else:
                    pieces.append(cache_indices[row_idx : row_idx + 1])
            return torch.cat(pieces, dim=0).to(dtype=torch.int32)

        outputs: list[torch.Tensor] = []
        for proj_states, conv_weights, bias, conv_state in (
            (q_proj_states, q_conv_weights, self.q_conv1d.bias, conv_state_q),
            (k_proj_states, k_conv_weights, self.k_conv1d.bias, conv_state_k),
            (v_proj_states, v_conv_weights, self.v_conv1d.bias, conv_state_v),
        ):
            conv_state[checkpoint_state_indices] = conv_state[source_state_indices]
            first = causal_conv1d_fn(
                self._select_segments_2d(proj_states, phase1_segments).transpose(0, 1),
                conv_weights,
                bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=has_initial_state[phase1_req_indices],
                cache_indices=build_cache_indices(phase1_segments),
                query_start_loc=self._make_cu(
                    [end - start for start, end, _, _ in phase1_segments],
                    device,
                ),
                metadata=None,
            ).transpose(0, 1)
            conv_state[source_state_indices] = conv_state[checkpoint_state_indices]
            second = causal_conv1d_fn(
                self._select_segments_2d(proj_states, phase2_segments).transpose(0, 1),
                conv_weights,
                bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=torch.ones(
                    len(phase2_segments), dtype=torch.bool, device=device
                ),
                cache_indices=cache_indices[checkpoint_row_tensor].to(
                    dtype=torch.int32
                ),
                query_start_loc=self._make_cu(
                    [end - start for start, end, _, _ in phase2_segments],
                    device,
                ),
                metadata=None,
            ).transpose(0, 1)
            output = torch.empty_like(proj_states)
            phase1_offset = 0
            for start, end, _, _ in phase1_segments:
                length = end - start
                output[start:end] = first[phase1_offset : phase1_offset + length]
                phase1_offset += length
            phase2_offset = 0
            for start, end, _, _ in phase2_segments:
                length = end - start
                output[start:end] = second[phase2_offset : phase2_offset + length]
                phase2_offset += length
            outputs.append(output)

        return outputs[0], outputs[1], outputs[2]
