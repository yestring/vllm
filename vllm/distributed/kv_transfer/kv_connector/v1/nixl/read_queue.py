# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Order ready READs while bounding how long short READs can defer long ones."""

from __future__ import annotations

from collections import OrderedDict


class ReadyReadQueue:
    """Keep FIFO order within each class in the worker's scheduling thread.

    Grant at most ``max_short_burst`` short READs while a long READ is waiting.
    A worker may requeue an unfinished long request at the tail after each chunk.
    """

    def __init__(self, max_short_burst: int) -> None:
        if type(max_short_burst) is not int or max_short_burst < 0:
            raise ValueError("max_short_burst must be a nonnegative integer")
        self._max_short_burst = max_short_burst
        self._short: OrderedDict[str, None] = OrderedDict()
        self._long: OrderedDict[str, None] = OrderedDict()
        self._short_grants = 0

    def add(self, request_id: str, *, short: bool) -> None:
        if request_id in self._short or request_id in self._long:
            raise ValueError(f"READ request is already ready: {request_id}")
        if short:
            self._short[request_id] = None
        else:
            if not self._long:
                self._short_grants = 0
            self._long[request_id] = None

    def pop_next(self) -> str | None:
        if self._long and (
            not self._short or self._short_grants >= self._max_short_burst
        ):
            self._short_grants = 0
            request_id, _ = self._long.popitem(last=False)
            return request_id
        if self._short:
            if self._long:
                self._short_grants += 1
            request_id, _ = self._short.popitem(last=False)
            return request_id
        return None

    def discard(self, request_id: str) -> None:
        self._short.pop(request_id, None)
        self._long.pop(request_id, None)
