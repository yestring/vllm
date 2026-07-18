# vllm/v1/core/kv_cache_policy.py
import time
import math
from typing import List
from vllm.v1.core.kv_cache_utils import KVCacheBlock

class BaseEvictionPolicy:
    def select_victims(self, candidates: List[KVCacheBlock], num_blocks: int) -> List[KVCacheBlock]:
        raise NotImplementedError

class LossAwareEvictionPolicy(BaseEvictionPolicy):
    """
    基于 Expected Eviction Loss 的驱逐策略。
    Loss = P(reuse) * RecomputeCost
    """
    def __init__(self, half_life_seconds: float = 60.0):
        self.half_life = half_life_seconds

    def _estimate_reuse_prob(self, block: KVCacheBlock) -> float:
        # 1. 使用 EMA 热度
        hotness = block.ema_hotness  # 0~1 之间，EMA 平滑
        
        # 2. 新近度因子（如果刚被访问过，reuse 概率飙升）
        if block.last_access_time > 0:
            delta = time.time() - block.last_access_time
            recency = math.exp(-delta / self.half_life)
        else:
            recency = 0.0
            
        # 3. 共享红利（共享人数越多，越不该踢）
        share_bonus = 1.0 + math.log(block.prefix_share_count + 1) * 0.2
        
        # 综合概率（加权组合）
        prob = (hotness * 0.6 + recency * 0.4) * share_bonus
        return min(prob, 1.0)

    def _estimate_recompute_cost(self, block: KVCacheBlock) -> float:
        # 基于 Token 数的重算成本（取对数防止长序列垄断保护）
        return math.log(block.num_tokens + 1) + 1.0

    def _compute_loss(self, block: KVCacheBlock) -> float:
        prob = self._estimate_reuse_prob(block)
        cost = self._estimate_recompute_cost(block)
        loss = prob * cost
        block.eviction_loss = loss
        return loss

    def select_victims(self, candidates: List[KVCacheBlock], num_blocks: int) -> List[KVCacheBlock]:
        if not candidates:
            return []
        # 批量计算 Loss
        for block in candidates:
            self._compute_loss(block)
        # 按 Loss 升序（Loss 最小的优先被驱逐）
        sorted_blocks = sorted(candidates, key=lambda b: b.eviction_loss)
        return sorted_blocks[:num_blocks]