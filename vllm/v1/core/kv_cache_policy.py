import math
import time
from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import KVCacheBlock


class BaseEvictionPolicy:
    def select_victims(
        self, candidates: Sequence[KVCacheBlock], num_blocks: int
    ) -> list[KVCacheBlock]:
        raise NotImplementedError


class LossAwareEvictionPolicy(BaseEvictionPolicy):
    """Choose blocks with the lowest expected eviction loss.

    Expected loss is estimated as reuse probability times the amount of prefix
    work invalidated by evicting the block.
    """

    def __init__(self, half_life_seconds: float = 60.0):
        self.half_life = half_life_seconds

    def _estimate_reuse_prob(
        self, block: KVCacheBlock, now: float
    ) -> float:
        if block.last_access_time > 0:
            delta = max(now - block.last_access_time, 0.0)
            recency = math.exp(-delta / self.half_life)
            decayed_frequency = block.access_frequency * recency
        else:
            recency = 0.0
            decayed_frequency = 0.0
        frequency_prob = 1.0 - math.exp(-decayed_frequency)
        return min(0.6 * frequency_prob + 0.4 * recency, 1.0)

    def _estimate_recompute_cost(self, block: KVCacheBlock) -> float:
        prefix_tokens = block.block_hash_num_tokens or block.num_tokens
        return math.log1p(prefix_tokens)

    def _compute_loss(self, block: KVCacheBlock, now: float) -> float:
        # Uncached blocks have no reusable content and should be consumed first.
        if block.block_hash is None:
            block.eviction_loss = -1.0
            return block.eviction_loss
        prob = self._estimate_reuse_prob(block, now)
        cost = self._estimate_recompute_cost(block)
        loss = prob * cost
        block.eviction_loss = loss
        return loss

    def select_victims(
        self, candidates: Sequence[KVCacheBlock], num_blocks: int
    ) -> list[KVCacheBlock]:
        if not candidates or num_blocks <= 0:
            return []
        now = time.time()
        sorted_blocks = sorted(
            candidates, key=lambda block: self._compute_loss(block, now)
        )
        return sorted_blocks[:num_blocks]
