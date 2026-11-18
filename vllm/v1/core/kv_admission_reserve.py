# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import deque


class KVAdmissionReserve:
    """Estimate one step of KV block growth from recent running batches."""

    def __init__(self, window_steps: int, quantile: float, max_blocks: int) -> None:
        if window_steps < 1:
            raise ValueError("KV admission reserve window must be positive")
        if not 0 < quantile <= 1:
            raise ValueError("KV admission reserve quantile must be in (0, 1]")
        if max_blocks < 0:
            raise ValueError("KV admission reserve block limit must be non-negative")
        self._samples: deque[int] = deque(maxlen=window_steps)
        self._quantile = quantile
        self._max_blocks = max_blocks
        self.blocks = 0

    def observe(self, growth_blocks: int) -> None:
        """Record net block growth before admission of waiting requests."""
        self._samples.append(min(max(growth_blocks, 0), self._max_blocks))
        ordered = sorted(self._samples)
        self.blocks = ordered[math.ceil(len(ordered) * self._quantile) - 1]
