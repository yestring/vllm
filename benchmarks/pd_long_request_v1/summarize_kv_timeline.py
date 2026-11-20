# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join P/D/NIXL timestamps to a vLLM timed-trace benchmark on one host."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median

STAGES = ("p_done", "d_forward", "read_submit", "read_done")
SHORT_TOKENS = 1024
LONG_TOKENS = 28 * 1024
CLIENT_ID = re.compile(r"(.+-)(0|[1-9][0-9]*)\Z")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield number, row


def nonnegative_int(value):
    return type(value) is int and value >= 0


def nonnegative_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def read_benchmark(case):
    trace_path = case / "trace.jsonl"
    trace = [row for _, row in read_jsonl(trace_path)]
    if not trace:
        raise ValueError(f"{trace_path}: empty trace")

    benchmark_path = case / "benchmark.json"
    with benchmark_path.open(encoding="utf-8") as stream:
        benchmark = json.load(stream)
    if not isinstance(benchmark, dict):
        raise ValueError(f"{benchmark_path}: expected a JSON object")
    for field in ("input_lens", "output_lens", "start_times", "ttfts", "errors"):
        column = benchmark.get(field)
        if not isinstance(column, list) or len(column) != len(trace):
            raise ValueError(f"{benchmark_path}: {field} must have {len(trace)} rows")

    for index, row in enumerate(trace):
        length = row.get("input_length")
        if not nonnegative_int(length) or length == 0:
            raise ValueError(f"{trace_path}:{index + 1}: invalid input_length")
        if error := benchmark["errors"][index]:
            raise ValueError(f"{benchmark_path}: request {index} failed: {error}")
        actual_length = benchmark["input_lens"][index]
        if not nonnegative_int(actual_length) or actual_length != length:
            raise ValueError(f"{benchmark_path}: prompt length mismatch for {index}")
        output_length = benchmark["output_lens"][index]
        if not nonnegative_int(output_length) or output_length == 0:
            raise ValueError(f"{benchmark_path}: request {index} returned no tokens")
        if not nonnegative_number(benchmark["ttfts"][index]):
            raise ValueError(f"{benchmark_path}: request {index} invalid TTFT")
        if not nonnegative_number(benchmark["start_times"][index]):
            raise ValueError(f"{benchmark_path}: request {index} invalid start_time")
    return trace, benchmark


def read_timeline(case, trace, benchmark):
    path = case / "timeline.jsonl"
    by_key = {}
    for line, event in read_jsonl(path):
        key, stage, timestamp = (event.get(k) for k in ("key", "stage", "ts_ns"))
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path}:{line}: missing request key")
        if stage not in STAGES or not nonnegative_int(timestamp):
            raise ValueError(f"{path}:{line}: invalid stage or ts_ns")
        events = by_key.setdefault(key, {})
        if stage in events:
            raise ValueError(f"{path}:{line}: duplicate {stage} for {key}")
        events[stage] = event

    prefixes = set()
    by_index = {}
    for key, events in by_key.items():
        missing = [stage for stage in STAGES if stage not in events]
        if missing:
            raise ValueError(f"{path}: {key} is missing {', '.join(missing)}")
        timestamps = [events[stage]["ts_ns"] for stage in STAGES]
        if timestamps != sorted(timestamps):
            raise ValueError(f"{path}: {key} has out-of-order stage timestamps")

        client_id = events["p_done"].get("client_request_id")
        match = CLIENT_ID.fullmatch(client_id) if isinstance(client_id, str) else None
        if match is None:
            raise ValueError(f"{path}: {key} has no indexed client_request_id")
        prefix, index_text = match.groups()
        index = int(index_text)
        prefixes.add(prefix)
        if len(prefixes) != 1:
            raise ValueError(f"{path}: mixed benchmark request ID prefixes")
        if index >= len(trace) or index in by_index:
            raise ValueError(f"{path}: invalid or duplicate benchmark index {index}")
        length = events["p_done"].get("prompt_tokens")
        if not nonnegative_int(length) or length != trace[index]["input_length"]:
            raise ValueError(
                f"{path}: {key} has prompt_tokens={length!r}, expected "
                f"{trace[index]['input_length']} for request {index}"
            )
        done = events["read_done"]
        for field in ("bytes", "descs", "xfer_ns"):
            if field in done and not nonnegative_int(done[field]):
                raise ValueError(f"{path}: {key} has invalid {field} on read_done")

        # The benchmark client and the four hooks use the same host's perf counter.
        start_seconds = benchmark["start_times"][index]
        start_ns = round(start_seconds * 1_000_000_000)
        if start_ns > events["p_done"]["ts_ns"]:
            raise ValueError(f"{path}: {key} has P response before client start")
        first_token_seconds = start_seconds + benchmark["ttfts"][index]
        first_token_ns = round(first_token_seconds * 1_000_000_000)
        if first_token_ns < done["ts_ns"]:
            raise ValueError(f"{path}: {key} has first token before READ completion")
        events["client_start"] = {"ts_ns": start_ns}
        events["first_token"] = {"ts_ns": first_token_ns}
        by_index[index] = events

    missing = [str(index) for index in range(len(trace)) if index not in by_index]
    if missing:
        raise ValueError(f"{path}: missing benchmark requests {', '.join(missing)}")
    return by_index


def ms_between(events, start, end):
    return (events[end]["ts_ns"] - events[start]["ts_ns"]) / 1_000_000


def print_case(case, trace, benchmark, requests):
    def show(value):
        return f"{value:.3f}" if value is not None else "—"

    print(f"\n{case}")
    print("#0 为预热。P 响应以代理收到为准，首 token 以客户端收到为准。单位 ms。")
    print(
        "| 请求 | 输入 token | 客户端 TTFT | 客户端发起 → P 响应 | P 响应 → D 转发 "
        "| D 转发 → READ 提交 | READ 提交 → 观察到完成 | READ 完成 → 首 token | P 响应 → 首 token "
        "| NIXL 传输耗时 | READ MiB | 描述符 |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    chronological = sorted(requests, key=lambda i: requests[i]["p_done"]["ts_ns"])
    for index in chronological:
        events = requests[index]
        done = events["read_done"]
        cells = [str(index), str(trace[index]["input_length"])]
        cells.append(show(benchmark["ttfts"][index] * 1000))
        cells.append(show(ms_between(events, "client_start", "p_done")))
        for start, end in zip(STAGES, (*STAGES[1:], "first_token")):
            cells.append(show(ms_between(events, start, end)))
        cells.append(show(ms_between(events, "p_done", "first_token")))
        cells.append(show(done["xfer_ns"] / 1_000_000 if "xfer_ns" in done else None))
        cells.append(show(done["bytes"] / 2**20 if "bytes" in done else None))
        cells.append(str(done.get("descs", "—")))
        print("| " + " | ".join(cells) + " |")

    shorts = [i for i in requests if i and trace[i]["input_length"] == SHORT_TOKENS]
    longs = [i for i in requests if i and trace[i]["input_length"] == LONG_TOKENS]
    for label, indices in (("1K", shorts), ("28K", longs)):
        if indices:
            ttft = median(benchmark["ttfts"][i] * 1000 for i in indices)
            read = median(
                ms_between(requests[i], "read_submit", "read_done") for i in indices
            )
            print(
                f"{label}（n={len(indices)}）客户端 TTFT 中位数 {ttft:.3f} ms，"
                f"READ 提交到观察到完成的中位数 {read:.3f} ms"
            )

    if not shorts or not longs:
        print("缺少 1K 或 28K 测量请求，无法检查两者是否存在重叠。")
        return
    overlap = []
    short_submitted_during_long = set()
    for short in shorts:
        short_start = requests[short]["read_submit"]["ts_ns"]
        short_end = requests[short]["read_done"]["ts_ns"]
        for long in longs:
            long_start = requests[long]["read_submit"]["ts_ns"]
            long_end = requests[long]["read_done"]["ts_ns"]
            if max(short_start, long_start) < min(short_end, long_end):
                overlap.append((short, long))
                if long_start <= short_start:
                    short_submitted_during_long.add(short)
    ratio = f"{len(short_submitted_during_long)}/{len(shorts)}"
    print(f"有 {ratio} 条 1K 请求在 28K READ 尚未被观察到完成时提交。")
    print(f"长短 READ 观察区间重叠的配对数 {len(overlap)}。")
    for short, long in overlap:
        print(f"  1K #{short} 和 28K #{long} 的 READ 观察区间重叠。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="+", metavar="CASE_DIR")
    args = parser.parse_args()
    for case in args.cases:
        try:
            trace, benchmark = read_benchmark(case)
            requests = read_timeline(case, trace, benchmark)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"{case}: {exc}\n")
        print_case(case, trace, benchmark, requests)


if __name__ == "__main__":
    main()
