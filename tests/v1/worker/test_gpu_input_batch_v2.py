# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Model Runner V2 request batching."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    _gather_kda_spec_workspace_metadata_kernel,
)

DEVICE = current_platform.device_type


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kda_spec_workspace_gpu_state_transition():
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    initialized = torch.tensor(
        [False, True, False, False], dtype=torch.bool, device="cuda"
    )
    active_batch = torch.tensor([True, False, False, False], device="cuda")
    active = torch.zeros(4, dtype=torch.int32, device="cuda")
    was_initialized = torch.zeros(4, dtype=torch.bool, device="cuda")
    slots_out = torch.empty(4, dtype=torch.int32, device="cuda")
    initialized_out = torch.empty(4, dtype=torch.bool, device="cuda")

    _gather_kda_spec_workspace_metadata_kernel[(1,)](
        idx_mapping,
        initialized,
        active_batch,
        active,
        was_initialized,
        slots_out,
        initialized_out,
        2,
        4,
        BLOCK_SIZE=8,
    )

    assert slots_out.cpu().tolist() == [3, 1, 0, 0]
    assert initialized_out.cpu().tolist() == [False, True, False, False]
    assert initialized.cpu().tolist() == [False, True, False, True]
    assert active.cpu().tolist() == [0, 0, 0, 1]
    assert was_initialized.cpu().tolist() == [False, True, False, False]
