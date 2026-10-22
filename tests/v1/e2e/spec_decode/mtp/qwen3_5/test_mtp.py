# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.distributed.conftest import MockSubscriber
from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ...utils import (
    assert_request_outputs_match,
    compute_acceptance_len,
    get_test_prompts,
)
from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
    ],
    ids=["qwen3_5-hybrid"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@single_gpu_only
def test_mtp_draft_lm_head_quantization(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """A quantized draft copy of the shared lm_head only changes drafting:
    greedy outputs match stock MTP and acceptance stays within 10%."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False)
    results = {}
    # NVFP4 needs a CUDA W4A16 kernel.
    quants = (None, "fp8", "nvfp4") if current_platform.is_cuda() else (None, "fp8")
    for quant in quants:
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B-Base",
            block_size=None,
            enable_chunked_prefill=None,
            max_model_len=2048,
            limit_mm_per_prompt={"image": 0, "video": 0},
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 3,
                "draft_lm_head_quantization": quant,
            },
            disable_log_stats=False,
        ) as runner:
            outputs = runner.llm.chat(prompts, sampling_config)
            results[quant] = (outputs, compute_acceptance_len(runner.llm.get_metrics()))

    ref_outputs, ref_al = results.pop(None)
    for quant, (outputs, al) in results.items():
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            # Not bit-exact: a different acceptance length changes the target's
            # verify batch shapes, which can flip a near-tied greedy token.
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context=f"draft_lm_head_quantization={quant}",
        )
        assert al >= 0.9 * ref_al, (
            f"{quant}: acceptance length {al:.3f} vs {ref_al:.3f}"
        )


@single_gpu_only
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="nvfp4 needs a CUDA W4A16 kernel"
)
def test_mtp_draft_lm_head_quantization_survives_sleep_and_reload(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    vllm_runner,
):
    """The quantized draft head is derived from the target head: it must come
    back intact after a level 1 sleep and be re-derived after a target weight
    reload, so outputs and acceptance match the run before."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = get_test_prompts(mm_enabled=False)
    with vllm_runner(
        "Qwen/Qwen3.5-0.8B-Base",
        block_size=None,
        enable_chunked_prefill=None,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "draft_lm_head_quantization": "nvfp4",
        },
        enable_sleep_mode=True,
        disable_log_stats=False,
    ) as runner:
        llm = runner.llm

        def run():
            before = llm.get_metrics()
            outputs = llm.chat(prompts, sampling_config)
            return outputs, compute_acceptance_len(llm.get_metrics(), before)

        ref_outputs, ref_al = run()

        llm.sleep(level=1)
        llm.wake_up()
        outputs, al = run()
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context="after level 1 sleep",
        )
        assert al == pytest.approx(ref_al, rel=0.03)

        llm.collective_rpc("reload_weights")
        outputs, al = run()
        assert_request_outputs_match(
            ref_outputs,
            outputs,
            required_matches=int(0.8 * len(ref_outputs)) + 1,
            context="after weight reload",
        )
        assert al == pytest.approx(ref_al, rel=0.03)


@single_gpu_only
def test_mtp_lookahead_kv_events(vllm_runner):
    """The public LLM path publishes dependencies separately from cached KV."""
    endpoint = f"tcp://127.0.0.1:{get_open_port()}"
    subscriber = MockSubscriber(endpoint, topic="", decode_type=KVEventBatch)
    tokens = list(range(1025))
    try:
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B-Base",
            block_size=None,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            enable_prefix_caching=True,
            prefix_cache_retention_interval=None,
            limit_mm_per_prompt={"image": 0, "video": 0},
            speculative_config={"method": "mtp", "num_speculative_tokens": 1},
            kv_events_config={
                "enable_kv_cache_events": True,
                "enable_lookahead_block_hashes": True,
                "endpoint": endpoint,
            },
        ) as runner:
            runner.llm.generate(
                {"prompt_token_ids": tokens},
                SamplingParams(temperature=0, max_tokens=1),
            )
            events: list[BlockStored] = []
            while (message := subscriber.receive_one(timeout=1000)) is not None:
                events.extend(
                    e for e in message[1].events if isinstance(e, BlockStored)
                )
            assert events
            for event in events:
                start = event.token_ids[0]
                assert event.token_ids == tokens[start : start + len(event.token_ids)]
                assert (
                    len(event.token_ids) == len(event.block_hashes) * event.block_size
                )
                assert event.lookahead_token_ids == [
                    tokens[end : end + 1]
                    for end in range(
                        start + event.block_size,
                        start + len(event.token_ids) + 1,
                        event.block_size,
                    )
                ]
                assert all(len(ids) == 1 for ids in event.lookahead_token_ids)
    finally:
        subscriber.close()
