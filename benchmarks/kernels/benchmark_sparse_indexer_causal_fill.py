# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark fused short-context sparse-indexer causal index generation."""

import argparse
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm.models.glm5next  # noqa: F401
from vllm.model_executor.layers.sparse_attn_indexer_kpool import (
    _fill_causal_indices,
)


def legacy_fill(rows: torch.Tensor, positions: torch.Tensor) -> None:
    causal_range = torch.arange(rows.shape[1], device=rows.device, dtype=torch.int32)
    positions = positions.to(torch.int32)
    rows[:] = causal_range[None, :]
    rows[causal_range[None, :] > positions[:, None]] = -1


def benchmark(fn, rows: torch.Tensor, positions: torch.Tensor) -> list[float]:
    return bench_gpu_time_with_cupti(
        fn,
        dry_run_time_ms=25,
        repeat_time_ms=200,
        use_cuda_graph=True,
        cold_l2_cache=True,
        input_args=(rows, positions),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 96, 512, 1536])
    parser.add_argument("--columns", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260828)
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(device)} torch={torch.__version__}")
    print("rows cols legacy_us fused_us speedup fused_ideal_GBps")

    for case_idx, num_rows in enumerate(args.rows):
        positions = torch.randint(
            0, args.columns, (num_rows,), device=device, dtype=torch.int64
        )
        legacy_rows = torch.empty(
            (num_rows, args.columns), device=device, dtype=torch.int32
        )
        fused_rows = torch.empty_like(legacy_rows)

        legacy_fill(legacy_rows, positions)
        _fill_causal_indices(fused_rows, positions)
        torch.testing.assert_close(fused_rows, legacy_rows)

        timings = {"legacy": [], "fused": []}
        functions = {
            "legacy": (legacy_fill, legacy_rows),
            "fused": (_fill_causal_indices, fused_rows),
        }
        order = ("legacy", "fused") if case_idx % 2 == 0 else ("fused", "legacy")
        for names in (order, reversed(order)):
            for name in names:
                fn, rows = functions[name]
                timings[name].extend(benchmark(fn, rows, positions))

        legacy_us = statistics.median(timings["legacy"]) * 1e3
        fused_us = statistics.median(timings["fused"]) * 1e3
        ideal_bytes = num_rows * (args.columns * 4 + positions.element_size())
        ideal_gbps = ideal_bytes / (fused_us * 1e-6) / 1e9
        print(
            f"{num_rows:4d} {args.columns:4d} {legacy_us:9.3f} "
            f"{fused_us:8.3f} {legacy_us / fused_us:7.2f}x "
            f"{ideal_gbps:16.2f}"
        )


if __name__ == "__main__":
    main()
