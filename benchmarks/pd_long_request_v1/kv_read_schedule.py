# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Schedule matched NIXL READs on one transport for the KV benchmark."""

import math
import time

SHORT_DESCS = 3_584
LONG_DESCS = 100_352
DESC_BYTES = 32 * 1024
DESCS_PER_MIB = 2**20 // DESC_BYTES
POLL_INTERVAL_NS = 250_000
CHUNK_POLICIES = ("chunk-fifo", "chunk-priority")
POLICIES = (
    "whole-overlap",
    *CHUNK_POLICIES,
    "serial",
    "isolated-short",
    "isolated-long",
)


def _summarize(chunks):
    if not chunks:
        return None
    submit_ns = chunks[0]["submit_ns"]
    done_ns = chunks[-1]["done_ns"]
    return {
        "submit_ns": submit_ns,
        "done_ns": done_ns,
        "elapsed_ns": done_ns - submit_ns,
        "nixl_ns": sum(chunk["nixl_ns"] for chunk in chunks),
        "bytes": sum(chunk["bytes"] for chunk in chunks),
        "descs": sum(chunk["descs"] for chunk in chunks),
        "posts": len(chunks),
        "chunks": chunks,
    }


def run_pair(transport, policy, chunk_mib, gap_ms, timeout_s):
    """Run one schedule and return a ledger using the local perf-counter clock.

    In paired concurrent modes, the short READ becomes eligible gap_ms after
    the first long post. In serial mode it becomes eligible at long completion.
    READ overlap is measured from host submits to observed completions.
    Chunks are half-open descriptor ranges.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if not math.isfinite(gap_ms) or gap_ms < 0:
        raise ValueError("gap_ms must be finite and nonnegative")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    chunk_descs = LONG_DESCS
    if policy in CHUNK_POLICIES:
        if not isinstance(chunk_mib, int) or chunk_mib <= 0:
            raise ValueError("chunk_mib must be a positive integer for chunk policies")
        chunk_descs = chunk_mib * DESCS_PER_MIB

    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns + int(timeout_s * 1e9)
    gap_ns = int(gap_ms * 1e6)
    long_end = SHORT_DESCS + LONG_DESCS
    long_next = long_end if policy == "isolated-short" else SHORT_DESCS
    want_short = policy != "isolated-long"
    short_ready_ns = start_ns if policy == "isolated-short" else None
    active = {}
    completed = {"long": [], "short": []}

    def post(role, start, count):
        handle, submit_ns = transport.post(start, count)
        active[role] = (handle, start, count, submit_ns)
        return submit_ns

    try:
        while active or long_next < long_end or (want_short and not completed["short"]):
            now_ns = time.perf_counter_ns()
            if now_ns >= deadline_ns:
                raise TimeoutError(f"{policy} exceeded {timeout_s:g} seconds")

            short_unposted = (
                want_short and not completed["short"] and "short" not in active
            )
            long_has_more = long_next < long_end
            if policy == "serial" and short_ready_ns is None and completed["long"]:
                short_ready_ns = completed["long"][-1]["done_ns"]
            short_allowed = (not active or policy == "whole-overlap") and (
                not long_has_more or policy != "chunk-fifo"
            )
            changed = False
            if (
                short_unposted
                and short_allowed
                and short_ready_ns is not None
                and short_ready_ns <= now_ns
            ):
                post("short", 0, SHORT_DESCS)
                changed = True
            elif not active and long_has_more:
                count = min(chunk_descs, long_end - long_next)
                submit_ns = post("long", long_next, count)
                if long_next == SHORT_DESCS and policy not in (
                    "serial", "isolated-long"
                ):
                    short_ready_ns = submit_ns + gap_ns
                long_next += count
                changed = True
                if policy == "whole-overlap" and gap_ns == 0:
                    # Give both READs a chance to be outstanding even at zero gap.
                    continue

            for role, (handle, start, count, submit_ns) in list(active.items()):
                stats = transport.check(handle)
                if stats is None:
                    continue
                del active[role]
                transport.release(handle)
                completed[role].append(
                    {**stats, "start": start, "count": count, "submit_ns": submit_ns}
                )
                changed = True
            if not changed:
                now_ns = time.perf_counter_ns()
                wake_ns = deadline_ns
                if active:
                    wake_ns = min(wake_ns, now_ns + POLL_INTERVAL_NS)
                if (
                    short_unposted
                    and short_ready_ns is not None
                    and short_ready_ns > now_ns
                ):
                    wake_ns = min(wake_ns, short_ready_ns)
                time.sleep(max(0, wake_ns - now_ns) / 1e9)
    finally:
        for handle, *_ in active.values():
            try:
                transport.release(handle)
            except Exception:
                # Keep the original failure. The transport owns final teardown.
                pass

    long = _summarize(completed["long"])
    short = _summarize(completed["short"])
    requests = [row for row in (long, short) if row is not None]
    last_done_ns = max(row["done_ns"] for row in requests)
    if last_done_ns > deadline_ns:
        raise TimeoutError(f"{policy} exceeded {timeout_s:g} seconds")
    elapsed_ns = last_done_ns - min(row["submit_ns"] for row in requests)
    return {
        "policy": policy,
        "chunk_mib": chunk_mib,
        "gap_ms": gap_ms,
        "short_ready_ns": short_ready_ns,
        "short_wait_ns": short["submit_ns"] - short_ready_ns if short else None,
        "short_response_ns": short["done_ns"] - short_ready_ns if short else None,
        "short_submitted_before_long_done": bool(
            short and long and short["submit_ns"] < long["done_ns"]
        ),
        "read_overlap": bool(
            short and any(
                short["submit_ns"] < chunk["done_ns"]
                and chunk["submit_ns"] < short["done_ns"]
                for chunk in completed["long"]
            )
        ),
        "elapsed_ns": elapsed_ns,
        "long": long,
        "short": short,
    }
