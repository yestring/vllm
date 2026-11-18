# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import isqrt


def _causal_attention_pairs(context_tokens: int, num_tokens: int) -> int:
    return num_tokens * context_tokens + num_tokens * (num_tokens + 1) // 2


class PrefillAttentionBudget:
    """Track causal query-key pairs scheduled in one prefill iteration."""

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("The prefill attention budget must be positive.")
        self._limit = limit
        self._remaining = limit

    def limit_chunk(
        self, context_tokens: int, num_tokens: int, *, reserve: int = 0
    ) -> int:
        """Fit a chunk after holding pairs for later requests in this iteration."""
        if num_tokens <= 0:
            return 0
        available = max(self._remaining - reserve, 0)
        if available < context_tokens + 1:
            # An undersized budget can still advance the first prefill.
            return int(self._remaining == self._limit)
        if _causal_attention_pairs(context_tokens, num_tokens) <= available:
            return num_tokens

        # Solve c * (c + 2 * context_tokens + 1) <= 2 * available.
        linear_term = 2 * context_tokens + 1
        discriminant = linear_term**2 + 8 * available
        return (isqrt(discriminant) - linear_term) // 2

    def consume(self, context_tokens: int, num_tokens: int) -> None:
        """Debit the chunk accepted by the scheduler and KV allocator."""
        self._remaining -= _causal_attention_pairs(context_tokens, num_tokens)

    def refund(self, context_tokens: int, num_tokens: int) -> None:
        """Restore the cost of a preempted chunk from this iteration."""
        self._remaining += _causal_attention_pairs(context_tokens, num_tokens)
