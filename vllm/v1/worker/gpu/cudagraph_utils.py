# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.offloader.base import get_offloader
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_aux_hidden_state_outputs: bool,
        device: torch.device,
        is_first_pp_rank: bool = True,
        use_pp: bool = False,
    ):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        self.device = device
        self.is_first_pp_rank = is_first_pp_rank
        self.use_pp = use_pp

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        self.uniform_decode_query_len = 1
        spec_config = vllm_config.speculative_config
        if spec_config is not None:
            self.uniform_decode_query_len += spec_config.num_speculative_tokens

        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = self.compilation_config.cudagraph_mode

        use_uniform_decode_cudagraph = (
            self.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.cudagraph_mode.separate_routine()
            and not self.use_pp
        )
        self.cudagraph_sizes, self.uniform_decode_cudagraph_sizes = get_cudagraph_sizes(
            self.compilation_config.cudagraph_capture_sizes,
            self.max_num_reqs,
            self.max_num_tokens,
            self.cudagraph_mode,
            self.uniform_decode_query_len,
            use_uniform_decode_cudagraph,
        )
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None
        if self.cudagraph_mode != CUDAGraphMode.NONE:
            self.pool = torch.cuda.graph_pool_handle()
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.intermediate_input_tensors: IntermediateTensors | None = None
        self.intermediate_output_tensors: IntermediateTensors | None = None
        self.intermediate_input_num_rows: dict[int, dict[str, int]] = {}
        self.intermediate_output_num_rows: dict[int, dict[str, int]] = {}

    def needs_capture(self) -> bool:
        return len(self.cudagraph_sizes) > 0

    def get_cudagraph_size(
        self, num_tokens: int, uniform_decode: bool = False
    ) -> int | None:
        if uniform_decode and self.uniform_decode_cudagraph_sizes:
            return self.uniform_decode_cudagraph_sizes.get(num_tokens)
        return self.cudagraph_sizes.get(num_tokens)

    @staticmethod
    def _slice_for_num_tokens(tensor: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor
        return tensor[: min(num_tokens, tensor.shape[0])]

    def _get_input_intermediate_tensors(
        self, model: nn.Module, num_tokens: int
    ) -> IntermediateTensors | None:
        return self.prepare_pp_intermediate_tensors(model, num_tokens, None)

    def prepare_pp_intermediate_tensors(
        self,
        model: nn.Module,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None:
        if self.is_first_pp_rank:
            return None
        if self.intermediate_input_tensors is None:
            self.intermediate_input_tensors = model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.vllm_config.model_config.dtype,
                device=self.device,
            )
            for v in self.intermediate_input_tensors.tensors.values():
                v.zero_()

        if intermediate_tensors is None:
            for v in self.intermediate_input_tensors.tensors.values():
                if v.ndim > 0:
                    v[:num_tokens].zero_()
        else:
            for key, src in intermediate_tensors.items():
                dst = self.intermediate_input_tensors[key]
                if src.ndim == 0:
                    dst.copy_(src, non_blocking=True)
                    continue
                copy_rows = min(src.shape[0], dst.shape[0], num_tokens)
                dst[:copy_rows].copy_(src[:copy_rows], non_blocking=True)
                if copy_rows < num_tokens:
                    dst[copy_rows:num_tokens].zero_()

        sliced_tensors = {
            k: self._slice_for_num_tokens(v, num_tokens)
            for k, v in self.intermediate_input_tensors.items()
        }
        self.intermediate_input_num_rows[num_tokens] = {
            k: (v.shape[0] if v.ndim > 0 else 0) for k, v in sliced_tensors.items()
        }
        return IntermediateTensors(sliced_tensors)

    def _copy_input_intermediate_tensors(
        self, num_tokens: int, intermediate_tensors: IntermediateTensors | None
    ) -> None:
        if self.is_first_pp_rank:
            return
        assert intermediate_tensors is not None
        assert self.intermediate_input_tensors is not None
        expected_rows = self.intermediate_input_num_rows.get(num_tokens)
        assert expected_rows is not None, (
            f"Missing PP intermediate input metadata for {num_tokens} tokens"
        )
        for key, num_rows in expected_rows.items():
            src = intermediate_tensors[key]
            dst = self.intermediate_input_tensors[key]
            if src.ndim == 0:
                dst.copy_(src, non_blocking=True)
                continue
            copy_rows = min(src.shape[0], num_rows, dst.shape[0])
            dst[:copy_rows].copy_(src[:copy_rows], non_blocking=True)
            if copy_rows < num_rows:
                dst[copy_rows:num_rows].zero_()

    def _allocate_output_buffers(
        self,
        model_output: torch.Tensor | IntermediateTensors,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        if isinstance(model_output, IntermediateTensors):
            if self.intermediate_output_tensors is None:
                self.intermediate_output_tensors = IntermediateTensors(
                    {k: torch.empty_like(v) for k, v in model_output.items()}
                )
        else:
            if self.hidden_states is None:
                self.hidden_states = torch.empty_like(model_output)
            if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                assert aux_hidden_states is not None
                self.aux_hidden_states = [
                    torch.empty_like(x) for x in aux_hidden_states
                ]

    def _copy_output_buffers(
        self,
        num_tokens: int,
        model_output: torch.Tensor | IntermediateTensors,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        if isinstance(model_output, IntermediateTensors):
            assert self.intermediate_output_tensors is not None
            num_rows = {}
            for k, v in model_output.items():
                if v.ndim == 0:
                    self.intermediate_output_tensors[k].copy_(v)
                    num_rows[k] = 0
                else:
                    self.intermediate_output_tensors[k][: v.shape[0]].copy_(v)
                    num_rows[k] = v.shape[0]
            self.intermediate_output_num_rows[num_tokens] = num_rows
            return

        assert self.hidden_states is not None
        self.hidden_states[:num_tokens].copy_(model_output)
        if self.use_aux_hidden_state_outputs:
            assert aux_hidden_states is not None
            for i, aux_hidden in enumerate(aux_hidden_states):
                self.aux_hidden_states[i][:num_tokens].copy_(aux_hidden)

    def capture_graph(
        self,
        num_tokens: int,
        capture_cg_mode: CUDAGraphMode,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        uniform_decode: bool = False,
    ) -> None:
        # select and check capture function
        assert capture_cg_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid capture_cudagraph_mode for capture: {capture_cg_mode}"
        )
        if capture_cg_mode == CUDAGraphMode.PIECEWISE:
            capture_fn = self._capture_piecewise_graph
        else:
            capture_fn = self._capture_full_graph
        # prepare inputs
        if uniform_decode:
            num_reqs = min(
                cdiv(num_tokens, self.uniform_decode_query_len),
                self.max_num_reqs,
            )
        else:
            num_reqs = min(num_tokens, self.max_num_reqs)

        model_inputs = {
            "input_ids": input_buffers.input_ids[:num_tokens],
            "positions": input_buffers.positions[:num_tokens],
            # NOTE: Values returned by `prepare_dummy_inputs` will override the
            # default values above.
            **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
        }
        if not self.is_first_pp_rank:
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None
            model_inputs["intermediate_tensors"] = self._get_input_intermediate_tensors(
                model, num_tokens
            )

        attn_metadata, slot_mappings = prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            model_state,
            input_buffers,
            block_tables,
            attn_groups,
            kv_cache_config,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)

        # Warm up.
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
        ):
            model_output = model(**model_inputs)
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

        # Allocate output buffers if not already done.
        self._allocate_output_buffers(hidden_states, aux_hidden_states)

        capture_fn(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            model=model,
            model_inputs=model_inputs,
            num_tokens_across_dp=num_tokens_across_dp,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            has_lora=has_lora,
        )

    def _capture_full_graph(
        self,
        num_tokens: int,
        num_reqs: int,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor | IntermediateTensors | None],
        num_tokens_across_dp: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        has_lora: bool = False,
    ) -> None:
        assert attn_metadata is not None
        # Capture the graph.
        assert num_tokens not in self.graphs
        graph = torch.cuda.CUDAGraph()

        # Sync offloader's copy stream before capture.
        # Ensure any pre-capture prefetches from offloader are complete.
        get_offloader().sync_prev_onload()

        with (
            set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
            ),
            torch.cuda.graph(graph, self.pool),
        ):
            model_output = model(**model_inputs)

            # Join offloader's copy stream after forward to avoid unjoined
            # stream error. The last layer's start_prefetch forks copy_stream,
            # but wait_prefetch only happens in the next forward pass.
            get_offloader().join_after_forward()

            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            # Copy outputs to the output buffers.
            self._copy_output_buffers(num_tokens, hidden_states, aux_hidden_states)
        self.graphs[num_tokens] = graph

    def _capture_piecewise_graph(
        self,
        num_tokens: int,
        num_reqs: int,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor | IntermediateTensors | None],
        num_tokens_across_dp: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        has_lora: bool = False,
    ) -> None:
        # create batch descriptor for piecewise cudagraph dispatch key
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens, has_lora=has_lora)

        # Capture run - CUDAGraphWrapper inside torch.compile will auto capture.
        with set_forward_context(
            attn_metadata=None,  # piecewise no need attn_metadata
            vllm_config=self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings,
        ):
            model(**model_inputs)

    @torch.inference_mode()
    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
    ) -> None:
        common_kwargs = dict(
            device=self.device,
            capture_fn=self.capture_graph,
            model=model,
            model_state=model_state,
            input_buffers=input_buffers,
            block_tables=block_tables,
            attn_groups=attn_groups,
            kv_cache_config=kv_cache_config,
            has_lora=has_lora,
        )

        # Phase 1: Capture for mixed prefill-decode batches if needed.
        mixed_mode = self.cudagraph_mode.mixed_mode()
        if mixed_mode != CUDAGraphMode.NONE:
            capture_graphs(
                cudagraph_sizes=self.cudagraph_sizes,
                capture_cudagraph_mode=mixed_mode,
                desc=f"Capturing CUDA graphs (mixed, {mixed_mode.name})",
                uniform_decode=False,
                **common_kwargs,
            )

        # Phase 2: Capture FULL graphs for uniform decode batches if needed.
        # This is only needed if we use a separate routine for decode batches
        # and the decode_mode is FULL.
        if self.uniform_decode_cudagraph_sizes:
            capture_graphs(
                cudagraph_sizes=self.uniform_decode_cudagraph_sizes,
                capture_cudagraph_mode=CUDAGraphMode.FULL,
                desc="Capturing CUDA graphs (decode, FULL)",
                uniform_decode=True,
                **common_kwargs,
            )

    def get_cudagraph_runtime_mode(
        self, num_reqs: int, num_tokens: int, max_query_len: int
    ) -> tuple[CUDAGraphMode, int | None]:
        is_uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            num_tokens == max_query_len * num_reqs
        )

        cudagraph_size = self.get_cudagraph_size(num_tokens, is_uniform_decode)
        if cudagraph_size is None:
            cudagraph_mode = CUDAGraphMode.NONE
        elif is_uniform_decode:
            cudagraph_mode = self.cudagraph_mode.decode_mode()
        else:
            cudagraph_mode = self.cudagraph_mode.mixed_mode()

        if (
            cudagraph_mode == CUDAGraphMode.FULL
            and cudagraph_size is not None
            and cudagraph_size not in self.graphs
        ):
            # If graph wasn't captured yet, fall back to eager.
            # This might happen when the dummy run is called before capture.
            cudagraph_mode = CUDAGraphMode.NONE
            cudagraph_size = None
        # PP decode can still use cudagraph through PIECEWISE mode, while
        # avoiding FULL replay path edge cases across stages.
        if self.use_pp and cudagraph_mode == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.PIECEWISE
        return cudagraph_mode, cudagraph_size

    def run_fullgraph(
        self, num_tokens: int, intermediate_tensors: IntermediateTensors | None = None
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        assert num_tokens in self.graphs, f"No cudagraph for {num_tokens} tokens"
        self._copy_input_intermediate_tensors(num_tokens, intermediate_tensors)
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[num_tokens].replay()
        output_rows = self.intermediate_output_num_rows.get(num_tokens)
        if output_rows is not None:
            assert self.intermediate_output_tensors is not None
            return IntermediateTensors(
                {
                    k: (
                        self.intermediate_output_tensors[k]
                        if num_rows == 0
                        else self.intermediate_output_tensors[k][:num_rows]
                    )
                    for k, num_rows in output_rows.items()
                }
            )
        assert self.hidden_states is not None
        hidden_states = self.hidden_states[:num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[:num_tokens] for x in self.aux_hidden_states]


def get_cudagraph_sizes(
    capture_sizes: list[int] | None,
    max_num_reqs: int,
    max_num_tokens: int,
    cudagraph_mode: CUDAGraphMode,
    uniform_decode_query_len: int = 1,
    uniform_decode_cudagraph: bool = False,
) -> tuple[dict[int, int], dict[int, int]]:
    # Support both FULL and PIECEWISE cudagraph modes
    if cudagraph_mode == CUDAGraphMode.NONE:
        return {}, {}
    if not capture_sizes:
        return {}, {}

    capture_sizes = sorted(capture_sizes)
    if not capture_sizes:
        return {}, {}

    cudagraph_sizes: dict[int, int] = {}
    for i in range(1, capture_sizes[-1] + 1):
        for x in capture_sizes:
            if i <= x:
                cudagraph_sizes[i] = x
                break

    uniform_decode_cudagraph_sizes: dict[int, int] = {}
    if uniform_decode_cudagraph:
        max_num_tokens = max_num_reqs * uniform_decode_query_len
        uniform_decode_cudagraph_sizes = {
            k: v
            for k, v in cudagraph_sizes.items()
            if v <= max_num_tokens and v >= uniform_decode_query_len
        }
    return cudagraph_sizes, uniform_decode_cudagraph_sizes


def capture_graphs(
    cudagraph_sizes: dict[int, int],
    device: torch.device,
    capture_fn: Callable,
    capture_cudagraph_mode: CUDAGraphMode,
    desc: str = "Capturing CUDA graphs",
    **capture_kwargs,
) -> None:
    # Capture larger graphs first.
    sizes_to_capture = sorted(set(cudagraph_sizes.values()), reverse=True)
    if is_global_first_rank():
        sizes_to_capture = tqdm(sizes_to_capture, desc=desc)

    with graph_capture(device=device):
        for size in sizes_to_capture:
            capture_fn(size, capture_cudagraph_mode, **capture_kwargs)


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    attn_metadata = model_state.prepare_attn(
        input_batch,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
    )
    return attn_metadata, slot_mappings_by_layer
