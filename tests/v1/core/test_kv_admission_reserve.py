# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_admission_reserve import KVAdmissionReserve
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _manager(window_steps: int) -> KVCacheManager:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    return KVCacheManager(
        kv_cache_config=KVCacheConfig(9, [], [KVCacheGroupSpec(["layer"], spec)]),
        max_model_len=128,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
        watermark=0.2,
        admission_reserve_window_steps=window_steps,
    )


def _request(prompt_blocks: int) -> Request:
    return Request(
        request_id="request",
        prompt_token_ids=[1] * (prompt_blocks * BLOCK_SIZE),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )


def test_reserve_tracks_nearest_rank_and_expires_old_growth():
    reserve = KVAdmissionReserve(window_steps=4, quantile=0.75, max_blocks=8)
    for sample in (-5, 2, 4, 20):
        reserve.observe(sample)
    assert reserve.blocks == 4
    for _ in range(4):
        reserve.observe(0)
    assert reserve.blocks == 0


def test_admission_reserve_only_applies_with_running_work():
    manager = _manager(window_steps=4)
    manager.update_admission_reserve(4)
    request = _request(prompt_blocks=5)
    assert manager.admission_watermark_blocks == 4
    assert (
        manager.allocate_slots(request, BLOCK_SIZE, full_sequence_must_fit=True) is None
    )
    assert manager.allocate_slots(
        request, BLOCK_SIZE, full_sequence_must_fit=True, has_scheduled_reqs=False
    ) is not None
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = BLOCK_SIZE
    assert manager.allocate_slots(request, 4 * BLOCK_SIZE) is not None


def test_admission_reserve_adds_to_inflight_protection_and_has_static_floor():
    manager = _manager(window_steps=1)
    manager.update_admission_reserve(4)
    request = _request(prompt_blocks=3)
    assert manager.allocate_slots(request, 3 * BLOCK_SIZE, reserved_blocks=2) is None
    manager.update_admission_reserve(0)
    assert manager.admission_watermark_blocks == 1
    assert manager.allocate_slots(
        request, 3 * BLOCK_SIZE, reserved_blocks=2
    ) is not None


def test_disabled_reserve_keeps_static_watermark():
    manager = _manager(window_steps=0)
    manager.update_admission_reserve(8)
    assert manager.admission_watermark_blocks == 1
    assert manager.allocate_slots(_request(7), 7 * BLOCK_SIZE) is not None
