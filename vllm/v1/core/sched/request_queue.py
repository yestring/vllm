# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)



# vllm/v1/core/sched/request_queue.py

class UtilityRequestQueue(RequestQueue):
    """
    Utility-driven request queue.

    Orders requests by a dynamically computed utility score.
    Since utility depends on waiting time (which changes every step),
    we reorder the entire list on each pop/peek operation.

    The utility is computed via `Request.utility_score(weights, now)`.
    """

    def __init__(self, utility_weights: dict[str, float] | None = None):
        """
        Args:
            utility_weights: dictionary with keys like 'wait', 'priority',
                'short', 'fair', 'mm_penalty' and their corresponding weights.
        """
        self._requests: list[Request] = []
        self._weights = utility_weights or {}

    def _reorder(self) -> None:
        """Reorder the internal list by utility descending."""
        if not self._requests:
            return
        now = time.monotonic()
        self._requests.sort(
            key=lambda req: req.utility_score(self._weights, now),
            reverse=True
        )

    def add_request(self, request: Request) -> None:
        """Add a request to the queue."""
        self._requests.append(request)

    def pop_request(self) -> Request:
        """Pop the request with the highest utility."""
        if not self._requests:
            raise IndexError("pop from empty utility queue")
        self._reorder()
        return self._requests.pop(0)

    def peek_request(self) -> Request:
        """Peek at the request with the highest utility without removing it."""
        if not self._requests:
            raise IndexError("peek from empty utility queue")
        self._reorder()
        return self._requests[0]

    def prepend_request(self, request: Request) -> None:
        """
        Prepend a request to the front of the queue.
        In utility ordering, "prepend" has no strict meaning because
        ordering is determined by utility. We simply add it to the list;
        the next reorder will place it correctly.
        """
        self._requests.append(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue (same as add)."""
        for req in requests:
            self._requests.append(req)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        try:
            self._requests.remove(request)
        except ValueError:
            # Request not found; silently ignore
            pass

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._requests = [req for req in self._requests if req not in requests_to_remove]

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        """
        Iterate over requests in utility order (highest first) without
        modifying the queue.
        """
        if not self._requests:
            return iter([])
        now = time.monotonic()
        sorted_copy = sorted(
            self._requests,
            key=lambda req: req.utility_score(self._weights, now),
            reverse=True
        )
        return iter(sorted_copy)




def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.UTILITY:
        # You can pass utility_weights via a global config; here we use defaults.
        # In practice, you'd want to read weights from SchedulerConfig.
        return UtilityRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
