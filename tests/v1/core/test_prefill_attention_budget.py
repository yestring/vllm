# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.core.sched.prefill_attention_budget import PrefillAttentionBudget


def test_chunk_consumes_budget_only_after_allocation():
    budget = PrefillAttentionBudget(10)
    assert budget.limit_chunk(3, 6) == 2
    assert budget.limit_chunk(3, 6) == 2
    budget.consume(3, 2)

    assert budget.limit_chunk(3, 6) == 0
    assert budget.limit_chunk(0, 6) == 1
    budget.consume(0, 1)
    assert budget.limit_chunk(0, 1) == 0


def test_too_small_budget_preserves_first_token_progress():
    budget = PrefillAttentionBudget(2)
    assert budget.limit_chunk(100, 0) == 0
    assert budget.limit_chunk(100, 32) == 1
    budget.consume(100, 1)
    assert budget.limit_chunk(0, 32) == 0


def test_reserve_leaves_attention_pairs_for_later_requests():
    budget = PrefillAttentionBudget(30)
    assert budget.limit_chunk(0, 16, reserve=20) == 4
    budget.consume(0, 4)

    assert budget.limit_chunk(0, 16, reserve=20) == 0
    assert budget.limit_chunk(0, 16) == 5


def test_refund_restores_budget_after_escape_hatch():
    budget = PrefillAttentionBudget(2)
    assert budget.limit_chunk(100, 32, reserve=1) == 1
    budget.consume(100, 1)
    assert budget.limit_chunk(0, 32, reserve=1) == 0
    budget.refund(100, 1)
    assert budget.limit_chunk(100, 32, reserve=1) == 1


def test_chunk_calculation_is_exact_on_large_integer_boundaries():
    context_tokens = 2**53
    target = 42
    cost = target * context_tokens + sum(range(1, target + 1))
    assert PrefillAttentionBudget(cost).limit_chunk(context_tokens, 64) == target
    assert PrefillAttentionBudget(cost - 1).limit_chunk(context_tokens, 64) == target - 1


def test_chunk_calculation_matches_brute_force():
    for context_tokens in (0, 1, 5, 20, 64):
        for limit in range(1, 100):
            budget = PrefillAttentionBudget(limit)
            for requested in range(1, 20):
                expected = 1
                for chunk in range(1, requested + 1):
                    end = context_tokens + chunk + 1
                    cost = sum(range(context_tokens + 1, end))
                    if cost > limit:
                        break
                    expected = chunk
                assert budget.limit_chunk(context_tokens, requested) == expected
