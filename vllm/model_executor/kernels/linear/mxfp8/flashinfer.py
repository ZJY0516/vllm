# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def _check_mm_mxfp8_shape(weight_shape: tuple[int, int]) -> tuple[bool, str | None]:
    N, K = weight_shape
    if K < 128:
        return False, f"mm_mxfp8 requires K >= 128, got K={K}."
    if K % MXFP8_BLOCK_SIZE != 0:
        return (
            False,
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}.",
        )
    if N < 128:
        return False, f"mm_mxfp8 requires N >= 128, got N={N}."
    return True, None


class FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CUTLASS (SM100+)."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda() and current_platform.has_device_capability(100)
        ):
            return False, "requires >=sm_100 (Blackwell)"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        can_implement, reason = _check_mm_mxfp8_shape(c.weight_shape)
        if not can_implement:
            return False, reason
        N, _ = c.weight_shape
        if current_platform.is_device_capability_family(120) and N % 32 != 0:
            return (
                False,
                f"mm_mxfp8 on SM12x requires N to be divisible by 32, got N={N}.",
            )
        return True, None

    def input_quant_key(self) -> QuantKey:
        # Activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        replace_parameter(layer, "weight", weight.contiguous())
        replace_parameter(layer, "weight_scale", weight_scale_swizzled.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        N, K = weight.shape

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = layer.params_dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                x.view(-1, K), is_sf_swizzled_layout=True
            )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cutlass",
        )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CuTe-DSL (SM100/SM103)."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires sm_100/sm_103 (Blackwell)"
        if not has_flashinfer_cutedsl():
            return False, "requires FlashInfer CuTe-DSL module"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return _check_mm_mxfp8_shape(c.weight_shape)

    def input_quant_key(self) -> QuantKey:
        # Activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        # Store weight column-major [K, N] as mm_mxfp8 expects for operand B.
        replace_parameter(layer, "weight", weight.contiguous().t())
        replace_parameter(layer, "weight_scale", weight_scale_swizzled.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # [K, N], column-major
        weight_scale = layer.weight_scale
        K, N = weight.shape

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = layer.params_dtype, x.shape
            x2 = x.view(-1, K)
            fast = _k31_fast_quant(x2)  # opt-in: K31_MXFP8_FASTQ=1
            if fast is not None:
                input_mxfp8, input_scale = fast
            else:
                input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                    x2, is_sf_swizzled_layout=True
                )

        output = _k31_tactic_mm(input_mxfp8, weight, input_scale, weight_scale, out_dtype)
        if output is None:
            output = vllm_flashinfer.mm_mxfp8(
                input_mxfp8,
                weight,
                input_scale,
                weight_scale,
                out_dtype=out_dtype,
                backend="cute-dsl",
            )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


# Opt-in (K31_MXFP8_TACTICS=1): fixed cute-dsl tactics for the Kimi K3.1 NVFP4
# checkpoint's MXFP8 linears at decode rows, picked by timing every valid tactic with
# cold weights (the untuned default and the autotuner, which profiles one hot weight,
# pick slower ones there). Output equals the default tactic's bit for bit at the timed
# rows. {(N, K): [(max rows, tactic), ...]}, applied for 33 <= rows <= 80.
# Opt-in (K31_MXFP8_FASTQ=1): decode-row activation quantization by k31_mxfp8q (the CuTe-DSL quantizer's
# arithmetic, laid out over the real rows); checked bitwise against mxfp8_e4m3_quantize on the first eager call per
# width, which falls back to the stock quantizer on any difference.
_K31_FASTQ = __import__("os").environ.get("K31_MXFP8_FASTQ") == "1"
_K31_FASTQ_OK: dict = {}


def _k31_fast_quant(x2):
    if not _K31_FASTQ or x2.dtype != torch.bfloat16 or not x2.is_contiguous():
        return None
    rows, width = x2.shape
    if width % 128 or not 0 < rows <= 256 or x2.data_ptr() % 16:
        return None
    from vllm.model_executor.layers.fused_moe import k31_mxfp8q

    ok = _K31_FASTQ_OK.get(width)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        q0, s0 = mxfp8_e4m3_quantize(x2, is_sf_swizzled_layout=True)
        q1, s1 = k31_mxfp8q.mxfp8_quant(x2)
        ok = torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)) and torch.equal(
            s0.view(torch.uint8), s1.view(torch.uint8)
        )
        _K31_FASTQ_OK[width] = ok
        from vllm.logger import init_logger

        init_logger(__name__).info("K31_MXFP8_FASTQ: K=%d rows=%d %s", width, rows,
                    "bitwise equal, on" if ok else "differs, off")  # fmt: skip
        return q0, s0
    return k31_mxfp8q.mxfp8_quant(x2) if ok else None


_K31_TACTICS = {
    (7168, 12288): [(80, ((256, 64), (2, 2), True, True, 1))],  # KDA o_proj
    (49376, 7168): [(64, ((256, 64), (4, 1), True, False, 1)),  # KDA in_proj
                    (80, ((256, 128), (2, 1), True, False, 1))],
    (12288, 7168): [(64, ((256, 64), (2, 1), True, False, 1)),  # shared gate_up
                    (80, ((256, 128), (2, 1), True, False, 1))],
    (7168, 6144): [(80, ((128, 64), (1, 4), False, True, 1))],  # shared down
}
_k31_cache: dict = {}


def _k31_tactic_mm(a, b, a_descale, b_descale, out_dtype):
    import os

    if os.environ.get("K31_MXFP8_TACTICS") != "1":
        return None
    m = a.shape[0]
    K, N = b.shape
    plan = _K31_TACTICS.get((N, K))
    if plan is None or not 33 <= m <= 80:
        return None
    tactic = next((t for max_m, t in plan if m <= max_m), None)
    from flashinfer.gemm import gemm_base as gb
    from flashinfer.utils import get_compute_capability

    out = torch.empty((m, N), device=a.device, dtype=out_dtype)
    ws = gb._get_stream_cache_buf("mm_mxfp8_workspace", gb.DEFAULT_WORKSPACE_SIZE, a.device)
    inputs = [a, b, a_descale, b_descale, out_dtype, out, ws]
    key = (N, K, m, out_dtype)
    runner = _k31_cache.get(key, 0)
    if runner == 0:
        runner = None
        major, minor = get_compute_capability(a.device)
        for r in gb._cute_dsl_mxfp8_runners(major, minor, True, out_dtype, inputs):
            try:
                if tactic in list(r.get_valid_tactics(inputs, None)):
                    runner = r
                    break
            except Exception:  # noqa: BLE001
                continue
        _k31_cache[key] = runner
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "K31_MXFP8_TACTICS: N=%d K=%d rows=%d tactic=%s %s", N, K, m, tactic,
            "used" if runner is not None else "not valid, default kept")
    if runner is None:
        return None
    runner(inputs=inputs, tactic=tactic)
    return out


class FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires SM100-family GPU"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        if hasattr(layer, "_mxfp8_trtllm_output_size") and layer.weight_scale.ndim == 1:
            return

        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        if K % 256 != 0:
            raise ValueError(
                f"FlashInfer TRTLLM MXFP8 requires K to be divisible by 256, got K={K}."
            )

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        padded_n = ((N + 127) // 128) * 128
        if padded_n != N:
            padded_weight = weight.new_zeros((padded_n, K))
            padded_weight[:N] = weight
            weight = padded_weight

            padded_scale = weight_scale.new_zeros((padded_n, scale_k))
            padded_scale[:N] = weight_scale
            weight_scale = padded_scale
        else:
            weight = weight.contiguous()

        replace_parameter(
            layer,
            "weight",
            shuffle_matrix_a(weight, 128).reshape(padded_n, K),
        )
        replace_parameter(
            layer,
            "weight_scale",
            shuffle_matrix_sf_a(
                weight_scale,
                128,
                num_elts_per_sf=MXFP8_BLOCK_SIZE,
            ).reshape(-1),
        )
        layer._mxfp8_trtllm_output_size = N

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dtype == torch.bfloat16, (
            f"FlashInfer TRTLLM MXFP8 requires bfloat16 activations, got {x.dtype}."
        )

        weight = layer.weight  # shuffled [padded N, K]
        weight_scale = layer.weight_scale
        _, K = weight.shape
        output_size = layer._mxfp8_trtllm_output_size
        input_shape = x.shape
        input_2d = x.view(-1, K)

        input_mxfp8, input_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_8x4(
            input_2d
        )
        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=x.dtype,
            backend="trtllm",
            use_8x4_sf_layout=True,
        )
        if output.shape[-1] != output_size:
            output = output[:, :output_size].contiguous()

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_size)
        return output.view(output_shape)
