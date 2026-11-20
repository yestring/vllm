#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the native and segmented NIXL READ paths on a cold burst trace."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from workload import positive_float, prepare_trace, summarize, synthetic_trace

TRACE = Path("/workspace/vllm-benchmark.KNwXSi/matrix/cold.jsonl")
MODEL = "/workspace/models/Qwen3-0.6B"
BENCHMARK = Path("benchmarks/pd_long_request_v1/run_baseline.py")
MAX_MODEL_LEN = 32_768
OFFERED = 300
TRACE_RATE = 2
TTFT_MS = 2_500
TPOT_MS = 100
PREFILL_BUDGET = 8_388_608
READ_CHUNK_BYTES = 256 * 2**20
PHASES = (("before", 0, 75), ("burst", 75, 225), ("after", 225, 300))
DETAILED_COLUMNS = ("input_lens", "output_lens", "ttfts", "itls", "errors", "start_times")
PORT_FLAGS = (
    "prefill-port", "decode-port", "proxy-port", "prefill-side-port", "decode-side-port",
)
ARMS = (
    ("native", False, False),
    ("prefill-only", True, False),
    ("segmented-only", False, True),
    ("both", True, True),
)
REQUIRED_METRICS = ("nixl_xfer_time_seconds_count", "nixl_bytes_transferred_sum")
OPTIONAL_METRICS = ("nixl_xfer_time_seconds_sum", "nixl_num_failed_transfers_total")
SEGMENTED_LOG = "NIXL Pull segmented READ enabled"


def load_trace(path):
    rows, _ = prepare_trace(path, MAX_MODEL_LEN, 0, 0, False)
    expected = synthetic_trace("prefill", OFFERED, TRACE_RATE)
    shape = ("timestamp", "input_length", "output_length")
    if len(rows) != OFFERED or any(
        any(row[key] != reference[key] for key in shape)
        for row, reference in zip(rows, expected)
    ):
        raise ValueError(f"expected a 300-request, 2 req/s prefill burst trace at {path}")
    prefix_keys = {(row.get("cache_salt"), row["hash_ids"][0]) for row in rows}
    if len(prefix_keys) != OFFERED:
        raise ValueError(f"expected distinct first prefix cache keys at {path}")
    return rows


def allocate_ports(used):
    ports = []
    with ExitStack() as stack:
        while len(ports) < len(PORT_FLAGS):
            listener = stack.enter_context(socket.socket())
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            if port not in used:
                used.add(port)
                ports.append(port)
    return ports


def command_for(args, case, prefill, segmented, ports):
    command = [
        sys.executable, str(args.repo / BENCHMARK), "--repo", str(args.repo),
        "--trace", str(args.trace), "--output", str(case),
        "--mode", "pull", "--model", args.model, "--max-model-len", str(MAX_MODEL_LEN),
        "--prefill-gpu", args.prefill_gpu, "--decode-gpu", args.decode_gpu,
        "--time-scale", str(TRACE_RATE / args.rate), "--eager", "--seed", "0",
        "--ttft-ms", str(TTFT_MS), "--tpot-ms", str(TPOT_MS),
        "--startup-timeout", str(args.startup_timeout),
        "--benchmark-timeout", str(args.benchmark_timeout),
        "--prefill-arg=--enable-prefix-caching", "--decode-arg=--enable-prefix-caching",
    ]
    command.extend(f"--{flag}={port}" for flag, port in zip(PORT_FLAGS, ports))
    if prefill:
        command.append(f"--prefill-arg=--prefill-attention-budget={PREFILL_BUDGET}")
    if segmented:
        # The last full --kv-transfer-config value wins in the vLLM CLI.
        config = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer",
            "engine_id": "pd-baseline-decode",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {"read_chunk_bytes": READ_CHUNK_BYTES},
        }
        command.append("--decode-arg=--kv-transfer-config=" + json.dumps(config))
    return command


def execute(command, repo):
    child = subprocess.Popen(command, cwd=repo, start_new_session=True)
    try:
        return child.wait()
    except KeyboardInterrupt:
        # run_baseline.py handles SIGTERM and shuts down its service groups.
        child.terminate()
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def transfer_deltas(path):
    first = last = None
    samples = 0
    names = REQUIRED_METRICS + OPTIONAL_METRICS
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["role"] != "decode":
                continue
            values = None
            if "metrics" in record:
                values = {name: {} for name in names}
                for metric in record["metrics"]:
                    series, value = metric.rsplit(maxsplit=1)
                    name = series.split("{", 1)[0].removeprefix("vllm:")
                    if name in values:
                        values[name][series] = float(value)
            if samples == 0:
                first = values
            last = values
            samples += 1
    if samples < 2 or first is None or last is None:
        raise ValueError("missing first or final Decode metrics snapshot")

    deltas = {}
    for name in names:
        before, after = first[name], last[name]
        # A fresh engine may expose a histogram series only after its first READ.
        valid = after and (not before or before.keys() == after.keys()) and all(
            0 <= before.get(key, 0) <= after[key] < math.inf for key in after
        )
        delta = sum(after[key] - before.get(key, 0) for key in after) if valid else None
        if name in REQUIRED_METRICS and (delta is None or delta <= 0):
            raise ValueError(f"missing or nonpositive Decode metric delta for {name}")
        deltas[name] = delta
    return deltas


def arm_results(case, rows, segmented):
    with (case / "decode.log").open(encoding="utf-8", errors="replace") as log:
        enabled = any(SEGMENTED_LOG in line for line in log)
    if enabled != segmented:
        raise ValueError(f"unexpected segmented READ status in Decode log (enabled={enabled})")
    manifest = read_json(case / "manifest.json")
    identity = {"revision": manifest["revision"], "trace_sha256": manifest["trace_sha256"]}
    summary = read_json(case / "summary.json")
    if summary["benchmark_prompt_length_mismatches"]:
        raise ValueError("benchmark prompt lengths differ from the source trace")
    groups = summary["groups"]
    counts = tuple(groups[key]["offered"] for key in ("all", "short", "long_input"))
    if counts != (OFFERED, 250, 50):
        raise ValueError("summary offered counts differ from the source trace")
    detail = read_json(case / "benchmark.json")
    calculated = summarize(detail, rows, TTFT_MS, TPOT_MS)
    if any(summary[key] != value for key, value in calculated.items()):
        raise ValueError("summary does not match the detailed benchmark results")
    phases = {}
    for phase, begin, end in PHASES:
        selected = {key: detail[key][begin:end] for key in DETAILED_COLUMNS}
        selected["duration"] = detail["duration"]
        short = summarize(selected, rows[begin:end], TTFT_MS, TPOT_MS)["groups"]["short"]
        # The reported phases are defined by offered arrival order, not completion time.
        phases[phase] = {key: short[key] for key in ("offered", "qualified", "ttft_p95_ms")}
    return {
        "identity": identity, "summary": summary, "short_phases": phases,
        "decode_nixl_deltas": transfer_deltas(case / "metrics.jsonl"),
    }


def report(matrix):
    def fmt(value, spec):
        return "—" if value is None else format(value, spec)

    rate = matrix["rate_requests_per_s"]
    lines = [
        f"Replay rate {rate:g} req/s. SLO limits TTFT {TTFT_MS:,} ms and TPOT {TPOT_MS} ms.",
        "",
        "| 策略 | 完成 | 短 TTFT P95 (ms) | 长 TTFT P95 (ms) | 长输入 SLO "
        "| Goodput (req/s) | 突发前短 SLO | 突发中短 SLO | 突发后短 SLO "
        "| D READ 完成数 | D READ GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    failures = []
    bytes_by_arm = {}
    for run in matrix["runs"]:
        name = run["arm"]
        if "error" in run:
            failures.append(f"- {name} — {run['error']}. 参见 `{run['directory']}`")
            continue
        groups = run["summary"]["groups"]
        overall, short, long = groups["all"], groups["short"], groups["long_input"]
        phases = run["short_phases"]
        slo = " | ".join(
            f"{phases[phase]['qualified']}/{phases[phase]['offered']}"
            for phase, _, _ in PHASES
        )
        transfer = run["decode_nixl_deltas"]
        count = transfer["nixl_xfer_time_seconds_count"]
        size = transfer["nixl_bytes_transferred_sum"]
        if overall["completed_with_tokens"] == OFFERED:
            bytes_by_arm[name] = size
        goodput = overall["goodput_requests_per_s_including_drain"]
        lines.append(
            f"| {name} | {overall['completed_with_tokens']}/{OFFERED} | "
            f"{fmt(short['ttft_p95_ms'], ',.1f')} | {fmt(long['ttft_p95_ms'], ',.1f')} | "
            f"{long['slo_attainment']:.1%} | {goodput:.3f} | {slo} | {count:.0f} | "
            f"{size / 2**30:.3f} |"
        )
    if "native" in bytes_by_arm and len(bytes_by_arm) > 1:
        native_size = bytes_by_arm["native"]
        discrepancies = [name for name, size in bytes_by_arm.items() if size != native_size]
        lines.append("")
        if discrepancies:
            names = ", ".join(discrepancies)
            lines.append(f"{names} 均完成全部请求，D READ 总字节数与 native 有差异。")
        else:
            lines.append("完成全部请求的对照组与 native 的 D READ 总字节数一致。")
    if failures:
        lines.extend(("", "失败组", *failures))
    return "\n".join(lines) + "\n"


def save(output, matrix):
    for name, content in (
        ("matrix.json", json.dumps(matrix, indent=2) + "\n"),
        ("report.md", report(matrix)),
    ):
        pending = output / f".{name}.pending"
        pending.write_text(content, encoding="utf-8")
        pending.replace(output / name)


def run_matrix(args, rows):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    (output / "matrix.json").touch(exist_ok=False)
    matrix = {
        "rate_requests_per_s": args.rate,
        "trace": str(args.trace),
        "prefill_budget": PREFILL_BUDGET,
        "read_chunk_bytes": READ_CHUNK_BYTES,
        "runs": [],
    }
    save(output, matrix)
    print(f"Results directory: {output}", flush=True)
    used_ports = set()
    baseline = None
    try:
        for name, prefill, segmented in ARMS:
            case = output / name
            current = {"arm": name, "directory": name}
            matrix["runs"].append(current)
            print(f"Running {name}: {case}", flush=True)
            try:
                ports = allocate_ports(used_ports)
                command = command_for(args, case, prefill, segmented, ports)
                current["command"] = command
                status = execute(command, args.repo)
                current["exit_code"] = status
                if status:
                    raise ValueError(f"baseline runner exited with status {status}")
                results = arm_results(case, rows, segmented)
                if baseline is None:
                    baseline = results["identity"]
                elif results["identity"] != baseline:
                    raise ValueError("repository revision or trace SHA-256 differs across arms")
                current.update(results)
            except KeyboardInterrupt:
                current["error"] = "interrupted during this arm"
                raise
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                current["error"] = str(exc)
                print(f"{name} failed: {exc}; inspect {case}", file=sys.stderr)
                return 1
            finally:
                save(output, matrix)
    finally:
        print(report(matrix), end="")
        print(f"Report: {output / 'report.md'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="2")
    parser.add_argument("--rate", type=positive_float, default=0.5)
    parser.add_argument("--startup-timeout", type=positive_float, default=180)
    parser.add_argument("--benchmark-timeout", type=positive_float, default=1800)
    args = parser.parse_args()
    pair = sorted((args.prefill_gpu, args.decode_gpu))
    if pair[0] == pair[1] or any(not gpu or "," in gpu for gpu in pair):
        parser.error("specify two distinct single GPU IDs")
    args.repo, args.trace, args.output = (
        path.resolve() for path in (args.repo, args.trace, args.output)
    )
    try:
        if not (args.repo / BENCHMARK).is_file():
            raise ValueError(f"cannot find {args.repo / BENCHMARK}")
        digest = hashlib.sha256(json.dumps(pair).encode()).hexdigest()
        lock = Path(tempfile.gettempdir()) / f"vllm-segmented-kv-{digest}.lock"
        flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
        with os.fdopen(os.open(lock, flags, 0o600), "r+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"matrix already running on GPU pair {', '.join(pair)}") from exc
            return run_matrix(args, load_trace(args.trace))
    except KeyboardInterrupt:
        print(f"Interrupted. Partial results: {args.output}", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"{exc}\nResults directory: {args.output}", file=sys.stderr)
        return 1


def _interrupt(_signal, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())
