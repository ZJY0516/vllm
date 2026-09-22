# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.utils.torch_utils import PIN_MEMORY, np_to_pinned_tensor
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


@dataclass
class CausalConv1dMetadata:
    """Prefill launch metadata shared by users of the Triton causal-conv kernel."""

    nums_dict: dict[int, dict[str, Any]]
    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor


def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: torch.Tensor, *, device: torch.device
) -> CausalConv1dMetadata:
    # Needed for causal_conv1d. Use the CPU query_start_loc to avoid DtoH sync.
    assert query_start_loc_p_cpu.device.type == "cpu"
    seqlens = query_start_loc_p_cpu.diff()
    nums_dict: dict[int, dict[str, Any]] = {}
    batch_ptr = None
    token_chunk_offset_ptr = None
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = np_to_pinned_tensor(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32, pin_memory=PIN_MEMORY)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            # Update default value after class definition
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                assert token_chunk_offset_ptr is not None
                token_chunk_offset_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)

        assert batch_ptr is not None
        batch_ptr[0:mlist_len].copy_(mlist, non_blocking=True)
        assert token_chunk_offset_ptr is not None
        token_chunk_offset_ptr[0:mlist_len].copy_(offsetlist, non_blocking=True)
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr

    return CausalConv1dMetadata(nums_dict, batch_ptr, token_chunk_offset_ptr)
