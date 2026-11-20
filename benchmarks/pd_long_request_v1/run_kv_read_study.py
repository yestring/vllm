# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare whole and segmented READ admission on the same two CUDA devices."""

import argparse
import json
import math
import os
import time
from importlib.metadata import distributions
from pathlib import Path
from statistics import median

from kv_read_schedule import run_pair
from kv_read_transport import DESC_BYTES, LONG_DESCS, SHORT_DESCS, open_transport

ARMS = (
    ("serial", "serial", None),
    ("whole-overlap", "whole-overlap", None),
    ("fifo-256", "chunk-fifo", 256),
    ("priority-128", "chunk-priority", 128),
    ("priority-256", "chunk-priority", 256),
    ("priority-512", "chunk-priority", 512),
)
GAPS_MS = (25, 50, 100)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="2")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seconds-per-arm", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-pairs", type=int, default=12)
    parser.add_argument("--pair-timeout", type=int, default=180)
    args = parser.parse_args()
    if not all((args.prefill_gpu, args.decode_gpu)) or args.prefill_gpu == args.decode_gpu:
        parser.error("select two distinct CUDA device indices or UUIDs")
    if any("," in gpu for gpu in (args.prefill_gpu, args.decode_gpu)):
        parser.error("select exactly one CUDA device per role")
    if min(args.seconds_per_arm, args.rounds, args.min_pairs, args.pair_timeout) < 1:
        parser.error("duration, rounds, sample count and timeout must be positive")
    if args.num_threads < 0:
        parser.error("num-threads must be nonnegative")
    return args


def verify_payload(row):
    for role, descs in (("short", SHORT_DESCS), ("long", LONG_DESCS)):
        read = row[role]
        if read["bytes"] != descs * DESC_BYTES or read["descs"] != descs:
            raise RuntimeError(
                f"{role} unexpected payload: {read['bytes']} bytes, {read['descs']} "
                f"descriptors; expected {descs * DESC_BYTES} bytes, {descs} descriptors"
            )


def percentile95_ms(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1] / 1e6


def write_report(output, completed):
    lines = [
        "| 组别 | 轮次 | 样本对数 | 短 ready→DONE P50/P95 ms | 短 ready→提交 P95 ms "
        "| 长首次提交→DONE P50/P95 ms | READ 实际重叠 | 短先 DONE "
        "| 长 READ 次数/条 | 每对搬运率 MiB/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, round_, rows in completed:
        short = [row["short_response_ns"] for row in rows]
        waiting = [row["short_wait_ns"] for row in rows]
        long = [row["long"]["elapsed_ns"] for row in rows]
        overlap = sum(row["read_overlap"] for row in rows)
        short_first = sum(row["short"]["done_ns"] < row["long"]["done_ns"] for row in rows)
        n = len(rows)
        count = rows[0]["long"]["posts"]
        pair_seconds = sum(row["elapsed_ns"] for row in rows) / 1e9
        pair_mib = (SHORT_DESCS + LONG_DESCS) * DESC_BYTES / 2**20
        mib_per_second = pair_mib * n / pair_seconds
        lines.append(
            f"| {label} | {round_} | {n} | "
            f"{median(short) / 1e6:.1f}/{percentile95_ms(short):.1f} | "
            f"{percentile95_ms(waiting):.1f} | "
            f"{median(long) / 1e6:.1f}/{percentile95_ms(long):.1f} | "
            f"{overlap}/{n} | {short_first}/{n} | {count} | {mib_per_second:.1f} |"
        )
    report = output / "report.md"
    pending = output / "report.pending"
    pending.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pending.replace(report)
    return report


def measure_arm(transport, policy, chunk_mib, args, output_file):
    # Validate real copies before collecting timing. This also primes full READ.
    transport.reset_destination()
    prime = run_pair(transport, policy, chunk_mib, GAPS_MS[0], args.pair_timeout)
    verify_payload(prime)
    transport.verify()

    rows = []
    end = time.monotonic() + args.seconds_per_arm
    with output_file.open("x", encoding="utf-8", buffering=1) as stream:
        while (
            len(rows) < args.min_pairs
            or len(rows) % len(GAPS_MS)
            or time.monotonic() < end
        ):
            gap_ms = GAPS_MS[len(rows) % len(GAPS_MS)]
            row = run_pair(transport, policy, chunk_mib, gap_ms, args.pair_timeout)
            verify_payload(row)
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows.append(row)
    return rows


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    settings = {
        **vars(args),
        "output": str(output),
        "gaps_ms": GAPS_MS,
        "arms": ARMS,
        "desc_bytes": DESC_BYTES,
        "ucx": {key: os.environ.get(key) for key in ("UCX_TLS", "UCX_NET_DEVICES")},
        "nixl_packages": {
            dist.metadata["Name"]: dist.version
            for dist in distributions()
            if (dist.metadata.get("Name") or "").lower().startswith("nixl")
        },
    }
    (output / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    completed = []
    print(f"Results directory: {output}", flush=True)
    with open_transport(
        args.prefill_gpu, args.decode_gpu, args.num_threads, args.pair_timeout
    ) as transport:
        for round_ in range(1, args.rounds + 1):
            order = ARMS if round_ % 2 else reversed(ARMS)
            for label, policy, chunk in order:
                stem = f"r{round_}-{label}"
                print(f"Running {stem}", flush=True)
                rows = measure_arm(transport, policy, chunk, args, output / f"{stem}.jsonl")
                completed.append((label, round_, rows))
                report = write_report(output, completed)
                print(f"Finished {stem}: {len(rows)} pairs", flush=True)
    print(report.read_text(encoding="utf-8"), end="")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
