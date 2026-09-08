# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_policy import LossAwareEvictionPolicy
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)

pytestmark = pytest.mark.cpu_test


def _cached_block(block_id: int, prefix_tokens: int) -> KVCacheBlock:
    block = KVCacheBlock(block_id)
    block.set_block_hash(
        make_block_hash_with_group_id(
            BlockHash(sha256(str(block_id).encode())), 0
        ),
        num_tokens=prefix_tokens,
    )
    block.num_tokens = 16
    return block


def test_loss_aware_policy_selects_from_all_candidates():
    policy = LossAwareEvictionPolicy(half_life_seconds=60.0)
    cold = _cached_block(0, 16)
    hot = _cached_block(1, 128)
    hot.access_frequency = 5.0
    hot.last_access_time = time.time()
    uncached = KVCacheBlock(2)

    victims = policy.select_victims([hot, cold, uncached], 2)

    assert victims == [uncached, cold]


def test_eviction_metadata_is_content_scoped():
    block = _cached_block(0, 64)
    block.access_frequency = 3.0
    block.last_access_time = time.time()
    block.eviction_loss = 2.0
    block.num_tokens = 16

    block.reset_hash()
    block.reset_eviction_metadata()

    assert block.block_hash is None
    assert block.access_frequency == 0.0
    assert block.last_access_time == 0.0
    assert block.num_tokens == 0
    assert block.eviction_loss == float("inf")
