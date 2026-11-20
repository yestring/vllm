# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
    _Read,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.read_queue import ReadyReadQueue

pytestmark = pytest.mark.cpu_test


class _FakeNixl:
    def __init__(self):
        self.posted = []
        self.released = []
        self.state = "DONE"

    def make_prepped_xfer(self, op, local, local_ids, remote, remote_ids, notif_msg):
        assert op == "READ"
        assert list(remote_ids) == [desc + 100 for desc in local_ids]
        self.posted.append((list(local_ids), notif_msg))
        return len(self.posted)

    def transfer(self, handle):
        return "PROC"

    def check_xfer_state(self, handle):
        return self.state

    def get_xfer_telemetry(self, handle):
        count = len(self.posted[handle - 1][0])
        return SimpleNamespace(totalBytes=count * 32768, descCount=count, xferDuration=1)

    def release_xfer_handle(self, handle):
        assert handle not in self.released
        self.released.append(handle)


def _worker():
    worker = object.__new__(NixlPullConnectorWorker)
    worker._read_chunk_bytes = 65536
    worker._ready_reads = ReadyReadQueue(max_short_burst=1)
    worker._segmented_reads = {}
    worker._recving_transfers = defaultdict(list)
    worker._recving_metadata = {}
    worker._invalid_block_ids = Queue()
    worker._failed_recv_reqs = Queue()
    worker._is_hma_required = False
    worker.nixl_wrapper = _FakeNixl()
    worker.xfer_stats = SimpleNamespace(
        record_transfer=lambda _: None, record_failed_transfer=lambda: None
    )
    worker._log_failure = lambda *args, **kwargs: None
    return worker


def _enqueue(worker, req, descs):
    local = np.array(descs)
    worker._segmented_reads[req] = _Read(
        1, 2, local, local + 100, "p", 0, req, f"{req}:1".encode(), 2
    )
    worker._ready_reads.add(req, short=len(descs) <= 2)


def _poll(worker):
    done = worker._pop_done_transfers(worker._recving_transfers)
    failed = set()
    while not worker._failed_recv_reqs.empty():
        failed.add(worker._failed_recv_reqs.get_nowait())
    return worker._retire_finished_reads(done, failed)


def test_short_read_runs_at_boundary_and_only_final_segment_notifies():
    worker = _worker()
    _enqueue(worker, "long", [0, 1, 2, 3, 4])
    worker._post_next_segment()
    _enqueue(worker, "short", [8])
    worker._post_next_segment()
    assert worker.nixl_wrapper.posted == [([0, 1], b"")]
    assert _poll(worker) == set()

    worker._post_next_segment()
    assert _poll(worker) == {"short"}
    worker._post_next_segment()
    assert _poll(worker) == set()
    worker._post_next_segment()
    assert _poll(worker) == {"long"}
    assert not worker._segmented_reads
    assert worker.nixl_wrapper.posted == [
        ([0, 1], b""),
        ([8], b"short:1"),
        ([2, 3], b""),
        ([4], b"long:1"),
    ]
    assert worker.nixl_wrapper.released == [1, 2, 3, 4]


def test_failed_middle_segment_invalidates_and_abandons_remaining_blocks():
    worker = _worker()
    _enqueue(worker, "long", [0, 1, 2, 3, 4])
    worker._recving_metadata["long"] = SimpleNamespace(local_block_ids=([11, 12],))
    worker._post_next_segment()
    assert _poll(worker) == set()

    worker._post_next_segment()
    worker.nixl_wrapper.state = "ERR"
    assert _poll(worker) == {"long"}
    assert worker._invalid_block_ids.get_nowait() == {11, 12}
    assert worker._invalid_block_ids.empty()
    assert not worker._segmented_reads
    worker._post_next_segment()
    assert worker.nixl_wrapper.posted == [([0, 1], b""), ([2, 3], b"")]
    assert worker.nixl_wrapper.released == [1, 2]


def test_short_burst_is_bounded_while_long_is_ready():
    ready = ReadyReadQueue(max_short_burst=1)
    ready.add("long", short=False)
    ready.add("short-1", short=True)
    ready.add("short-2", short=True)
    assert [ready.pop_next(), ready.pop_next(), ready.pop_next()] == [
        "short-1", "long", "short-2"
    ]
