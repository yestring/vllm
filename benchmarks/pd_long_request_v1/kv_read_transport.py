# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL UCX READ transport for the single-host two-GPU study.

The producer runs in a spawned process. The caller owns the consumer CUDA
context and must enter open_transport before initializing CUDA. Each process
uses its own CUDA_VISIBLE_DEVICES mask, so the local buffer is always cuda:0.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass

from kv_read_schedule import DESC_BYTES, LONG_DESCS, SHORT_DESCS

STRIDE_BYTES = 2 * DESC_BYTES
TOTAL_DESCS = SHORT_DESCS + LONG_DESCS

_BODY_PATTERN = 0xAB
_RESET_PATTERN = 0xCD
_PENDING = ("PROC", "PEND")


def _cuda_buffer(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Initialize the benchmark transport before using CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible CUDA GPU with mask {gpu}")
    torch.cuda.set_device(0)
    buffer = torch.empty((TOTAL_DESCS, STRIDE_BYTES), dtype=torch.uint8, device="cuda:0")

    indices = torch.arange(TOTAL_DESCS, dtype=torch.int32, device=buffer.device)
    tags = torch.stack(
        (
            indices & 255,
            (indices >> 8) & 255,
            indices >> 16,
            torch.where(indices < SHORT_DESCS, 0x31, 0x71),
        ),
        dim=1,
    ).to(torch.uint8)
    return buffer, tags


def _register_buffer(role, buffer, num_threads):
    # Match the vLLM Pull connector's imports and UCX agent configuration.
    from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

    if NixlWrapper is None or nixl_agent_config is None:
        raise RuntimeError("NIXL and its agent configuration are required in this venv")
    config = nixl_agent_config(num_threads=num_threads, capture_telemetry=True)
    agent = NixlWrapper(f"kv-read-study-{role}-{os.getpid()}", config)
    registration = agent.register_memory(
        [(buffer.data_ptr(), buffer.nbytes, buffer.device.index, "")],
        mem_type="VRAM",
        backends=["UCX"],
    )
    if registration is None:
        raise RuntimeError(f"{role} VRAM registration failed")
    return agent, registration


def _provider_main(connection, gpu, num_threads):
    with connection:
        try:
            source, tags = _cuda_buffer(gpu)
            import torch

            source[:, :DESC_BYTES].fill_(_BODY_PATTERN)
            source[:, :4] = tags
            source[:, DESC_BYTES - 4 : DESC_BYTES] = tags
            torch.cuda.synchronize()
            agent, registration = _register_buffer("prefill", source, num_threads)
            try:
                connection.send(
                    ("ready", agent.get_agent_metadata(), source.data_ptr(), source.device.index)
                )
                while not connection.poll(0.001):
                    agent.get_new_notifs()
                if connection.recv() != "stop":
                    raise RuntimeError("Unexpected producer control message")
            finally:
                agent.deregister_memory(registration)
        except EOFError:
            # If the caller closed its control socket, there is no peer to serve.
            return
        except Exception:
            try:
                connection.send(("error", traceback.format_exc()))
            except OSError:
                pass
            raise


def _dlist(agent, peer, address, device_id):
    # Adjacent transfer rows are separated by an unused 32 KiB gap. This keeps
    # the requested descriptor count independent of NIXL's merge optimization.
    rows = [
        (address + slot * STRIDE_BYTES, DESC_BYTES, device_id)
        for slot in range(TOTAL_DESCS)
    ]
    return agent.prep_xfer_dlist(peer, agent.get_xfer_descs(rows, "VRAM"))


@dataclass(eq=False)
class _Read:
    native: object
    start: int
    count: int
    prepare_ns: int
    post_call_ns: int = 0
    done_ns: int | None = None
    result: dict | None = None


class ReadTransport:
    def __init__(self, agent, local, remote, destination, tags):
        self._agent = agent
        self._local = local
        self._remote = remote
        self._destination = destination
        self._tags = tags
        self._active = set()
        self._completed = set()

    def reset_destination(self):
        """Reset the receive buffer and verification state between trials."""
        import torch

        if self._active:
            raise RuntimeError("Cannot reset GPU memory while READ handles are live")
        self._destination[:, :DESC_BYTES].fill_(_RESET_PATTERN)
        torch.cuda.synchronize()
        self._completed.clear()

    def post(self, start, count):
        """Post a READ of count descriptors starting at start on both sides."""
        if start < 0 or count <= 0 or start + count > TOTAL_DESCS:
            raise ValueError(f"Invalid READ slot range [{start}, {start + count})")
        prepare_start = time.perf_counter_ns()
        indices = list(range(start, start + count))
        native = self._agent.make_prepped_xfer(
            "READ", self._local, indices, self._remote, indices
        )
        prepared = time.perf_counter_ns()
        handle = _Read(native, start, count, prepared - prepare_start)
        self._active.add(handle)
        submit_ns = time.perf_counter_ns()
        state = self._agent.transfer(native)
        posted_ns = time.perf_counter_ns()
        handle.post_call_ns = posted_ns - submit_ns
        if state == "DONE":
            handle.done_ns = posted_ns
        elif state not in _PENDING:
            raise RuntimeError(f"NIXL READ post returned {state!r} for slots {start}+{count}")
        return handle, submit_ns

    def check(self, handle):
        """Return completed READ timings, or None while it is in progress."""
        if handle not in self._active:
            raise ValueError("READ handle is not active")
        if handle.result is not None:
            return handle.result
        if handle.done_ns is None:
            state = self._agent.check_xfer_state(handle.native)
            if state in _PENDING:
                return None
            if state != "DONE":
                raise RuntimeError(f"NIXL READ failed for slots {handle.start}+{handle.count}: {state}")
            handle.done_ns = time.perf_counter_ns()

        telemetry = self._agent.get_xfer_telemetry(handle.native)
        size, count = int(telemetry.totalBytes), int(telemetry.descCount)
        if size != handle.count * DESC_BYTES or count != handle.count:
            raise RuntimeError(
                f"Unexpected NIXL telemetry for slots {handle.start}+{handle.count}: "
                f"{size} bytes and {count} descriptors"
            )
        handle.result = {
            "done_ns": handle.done_ns,
            "nixl_ns": int(telemetry.xferDuration * 1_000),
            "bytes": size,
            "descs": count,
            "prepare_ns": handle.prepare_ns,
            "post_call_ns": handle.post_call_ns,
            "nixl_post_ns": int(telemetry.postDuration * 1_000),
        }
        self._completed.add((handle.start, handle.start + handle.count))
        return handle.result

    def release(self, handle):
        """Release an observed completed transfer handle."""
        if handle not in self._active or handle.done_ns is None:
            raise ValueError("Only completed READ handles can be released")
        self._agent.release_xfer_handle(handle.native)
        self._active.remove(handle)

    def verify(self):
        """Check first, middle, and last sentinels on GPU for every selected slot."""
        import torch

        if self._active:
            raise RuntimeError("Release all READ handles before verifying GPU memory")
        if not self._completed:
            raise RuntimeError("No completed READ slots to verify")
        torch.cuda.synchronize()
        checks = []
        for start, end in self._completed:
            values = self._destination[start:end]
            tags = self._tags[start:end]
            checks.append(
                (values[:, :4] == tags).all()
                & (values[:, DESC_BYTES // 2] == _BODY_PATTERN).all()
                & (values[:, DESC_BYTES - 4 : DESC_BYTES] == tags).all()
            )
        if not torch.stack(checks).all().item():
            raise RuntimeError("READ destination sentinel mismatch")

    def _drain(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while self._active:
            for handle in tuple(self._active):
                if handle.done_ns is None:
                    state = self._agent.check_xfer_state(handle.native)
                    if state in _PENDING:
                        continue
                self._agent.release_xfer_handle(handle.native)
                self._active.remove(handle)
            if self._active:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out draining READ handles before cleanup")
                time.sleep(0.001)


def _stop_provider(process, connection, timeout_s):
    if process.pid is None:
        return
    if process.is_alive():
        try:
            connection.send("stop")
        except (EOFError, OSError):
            pass
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(1)
    if process.is_alive():
        process.kill()
        process.join(1)
    if process.exitcode != 0:
        raise RuntimeError(f"NIXL producer process exited with status {process.exitcode}")


@contextmanager
def open_transport(prefill_gpu, decode_gpu, num_threads, timeout_s):
    """Open a producer and a consumer using the active vLLM Python environment."""
    masks = (str(prefill_gpu), str(decode_gpu))
    if masks[0] == masks[1] or any(not mask or "," in mask for mask in masks):
        raise ValueError("Specify two distinct CUDA GPU indices or UUIDs")
    if num_threads < 0 or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("num_threads must be nonnegative and timeout_s must be positive")

    context = multiprocessing.get_context("spawn")
    connection, producer_connection = context.Pipe()
    producer = context.Process(
        target=_provider_main, args=(producer_connection, prefill_gpu, num_threads)
    )
    old_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    cleanup_timeout_s = min(5.0, timeout_s)
    failure = None
    with connection, producer_connection:
        try:
            producer.start()
            producer_connection.close()
            if not connection.poll(timeout_s):
                raise TimeoutError("Timed out waiting for producer NIXL metadata")
            try:
                status, *payload = connection.recv()
            except EOFError as exc:
                raise RuntimeError("NIXL producer exited before sending metadata") from exc
            if status != "ready":
                raise RuntimeError(f"NIXL producer setup failed:\n{payload[0]}")
            metadata, remote_address, remote_device = payload
            if not isinstance(metadata, bytes) or not metadata:
                raise RuntimeError("NIXL producer did not send native agent metadata")

            destination, tags = _cuda_buffer(decode_gpu)
            agent, registration = _register_buffer("decode", destination, num_threads)
            peer = local = remote = transport = consumer_error = None
            try:
                peer = agent.add_remote_agent(metadata)
                local = _dlist(
                    agent, "NIXL_INIT_AGENT", destination.data_ptr(), destination.device.index
                )
                remote = _dlist(agent, peer, remote_address, remote_device)
                transport = ReadTransport(agent, local, remote, destination, tags)
                yield transport
            except BaseException as exc:
                consumer_error = exc
                raise
            finally:
                try:
                    if producer.exitcode is not None:
                        raise RuntimeError("NIXL producer exited while consumer was active")
                    if transport is not None:
                        transport._drain(cleanup_timeout_s)
                    if remote is not None:
                        agent.release_dlist_handle(remote)
                    if local is not None:
                        agent.release_dlist_handle(local)
                    if peer is not None:
                        agent.remove_remote_agent(peer)
                    agent.deregister_memory(registration)
                except Exception as exc:
                    if consumer_error is None:
                        raise
                    consumer_error.add_note(f"NIXL consumer cleanup: {exc}")
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if old_mask is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_mask
            try:
                _stop_provider(producer, connection, cleanup_timeout_s)
            except Exception as exc:
                if failure is None:
                    raise
                failure.add_note(str(exc))
