# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K projection on the FlashInfer CuTe core vendored in `splitk.py` (with an
evict_first weight TMA; policies map padded rows to (M tile, N tile, split K,
stages); attention owns live slot masks)."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.host import architecture, tensor_maps, to_cute
from ..common.host import stream as current_stream
from .splitk import SplitKGemm, SplitKPushGemm, SplitKTactic


def matrix_view(tensor, *, output=False):
    """Keep packed row pitches and FlashInfer's singleton-safe dynamic major axis."""
    rows, cols = tensor.shape
    shape = (cols, rows, 1) if output else (rows, cols, 1)
    stride = (1, tensor.stride(0), 0) if output else (tensor.stride(0), 1, 0)
    result = to_cute(tensor.as_strided(shape, stride))
    return result.mark_layout_dynamic(leading_dim=0 if output else 1)


@cache
def compile_projection(tactic, output_dtype, arch):
    """Code-only cache; weights, descriptors and activation pointers stay private."""

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    y = fake(0, cutlass.Float32 if output_dtype == torch.float32 else cutlass.BFloat16)
    split = SplitKTactic(*tactic)
    kernel = SplitKGemm(tactic=split)
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


@cache
def compile_push_projection(tactic, arch, tp):
    """`SplitKPushGemm` (BF16 rows, also peer-stored into the MoE tail's RS
    mailboxes); args (w, x, y, a_map, peers [tp] Int64, parity, rank, capacity)."""

    def fake(stride):
        shape = tuple(cute.sym_int() for _ in range(3))
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    w, x = fake((cute.sym_int(64), 1, 0)), fake((cute.sym_int(64), 1, 0))
    y = fake((1, cute.sym_int(64), 0))
    peers = make_fake_tensor(cutlass.Int64, (tp,), (1,), assumed_align=16)
    kernel = SplitKPushGemm(tactic=SplitKTactic(*tactic), tp=tp)
    i32 = cutlass.Int32(0)
    args = (w, x, y, cutlass.Int64(0), peers, i32, i32, i32, make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


def weight_map(weight, rows):
    """Device TMA map of `weight` [N, K] (row pitch kept): box {64, rows}, SW128."""
    types = cuda.CUtensorMapDataType
    dtype = {
        torch.bfloat16: types.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        torch.float16: types.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    }[weight.dtype]
    swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    n, k = weight.shape
    pitch = weight.stride(0) * weight.element_size()
    spec = (weight.data_ptr(), dtype, (k, n), (pitch,), (64, rows), swizzle, l2)
    return tensor_maps([spec], weight.device)


class PreparedProjection:
    """Split-K projection with a per-batch tactic table and caller-owned outputs."""

    def __init__(self, weight, policies, *, output_dtype=torch.bfloat16):
        if weight.dim() != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"projection weight must be a BF16/FP16 [N, K] matrix, got "
                f"{tuple(weight.shape)} {weight.dtype}"
            )
        pitch = weight.stride(0) * weight.element_size()
        if weight.stride(1) != 1 or pitch % 16 or weight.data_ptr() % 16:
            raise ValueError(
                "projection weight rows need unit stride and a 16 B TMA pitch"
            )
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported projection output dtype {output_dtype}")
        self.weight, self.device = weight, weight.device
        self.output_dtype, self.policies = output_dtype, dict(policies)
        self.weight_view = matrix_view(weight)
        self.kernels, self.extra, self.maps = {}, {}, {}

    def prepare(self):
        arch, dtype = architecture(self.device), self.output_dtype
        with torch.accelerator.device_index(self.device.index):
            for batch, tactic in self.policies.items():
                code = compile_projection(tactic, dtype, arch)
                self.kernels[batch] = code
                self.extra[batch] = self.weight_map(tactic[0])
        return self

    def weight_map(self, rows):
        """The kernel's extra argument: A's map for `rows`-row tiles."""
        if rows not in self.maps:
            self.maps[rows] = weight_map(self.weight, rows)
        return (cutlass.Int64(self.maps[rows].data_ptr()),)

    def run(self, x, out):
        y = matrix_view(out, output=True)
        b, stream = x.shape[0], current_stream(self.device)
        if b not in self.kernels or x.shape[1] != self.weight.shape[1]:
            raise ValueError(
                f"projection input {tuple(x.shape)}: prepared batches "
                f"{sorted(self.kernels)}, K {self.weight.shape[1]}"
            )
        code, extra = self.kernels[b], self.extra[b]
        code(self.weight_view, matrix_view(x), y, y, *extra, stream)
        return out

    def push_kernel(self, batch, tp):
        """The peer-storing variant of `batch`'s tactic (BF16 output only)."""
        if self.output_dtype != torch.bfloat16:
            raise ValueError("push projection: BF16 output only")
        tactic = self.policies[batch]
        arch = architecture(self.device)
        with torch.accelerator.device_index(self.device.index):
            return compile_push_projection(tactic, arch, tp), self.weight_map(tactic[0])

    def run_push(self, code, x, out, peers, parity, rank, capacity):
        """`run` whose rows are also peer-stored (contiguous [B, 7168] rows)."""
        if out.stride(0) != 7168 or out.shape[1] != 7168 or out.stride(1) != 1:
            shape = (tuple(out.shape), out.stride())
            raise ValueError(f"push projection output {shape}")
        kernel, extra = code
        y = matrix_view(out, output=True)
        stream = current_stream(self.device)
        args = (self.weight_view, matrix_view(x), y, *extra, peers, parity, rank)
        kernel(*args, capacity, stream)
        return out
