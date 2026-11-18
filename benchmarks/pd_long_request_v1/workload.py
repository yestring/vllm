"""Prepare an open-loop workload and summarize vLLM detailed serving results."""

import argparse
import json
import math
from pathlib import Path


def write_trace(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def synthetic_trace(scenario, requests, rate):
    next_hash = 0
    for index in range(requests):
        input_len, output_len = 1024, 128
        in_burst = requests // 4 <= index < 3 * requests // 4
        if in_burst and index % 3 == 0:
            if scenario == "prefill":
                input_len = (8192, 16384, 28672)[index // 3 % 3]
            elif scenario == "decode":
                output_len = 2048
        blocks = math.ceil(input_len / 512)
        yield {
            "timestamp": round(1000 * index / rate),
            "input_length": input_len,
            "output_length": output_len,
            "hash_ids": list(range(next_hash, next_hash + blocks)),
        }
        next_hash += blocks


def prepare_trace(path, max_model_len, start, limit, drop_oversize):
    rows = []
    dropped = 0
    previous = -1
    origin = None
    with path.open(encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, 1):
            if line_num <= start:
                continue
            try:
                item = json.loads(line)
                timestamp = item["timestamp"]
                input_len, output_len = item["input_length"], item["output_length"]
                hashes = item["hash_ids"]
                if (
                    type(timestamp) not in (int, float)
                    or not math.isfinite(timestamp)
                    or timestamp < 0
                    or timestamp < previous
                    or type(input_len) is not int
                    or type(output_len) is not int
                    or input_len < 1
                    or output_len < 1
                    or not isinstance(hashes, list)
                    or not hashes
                    or any(type(block) is not int for block in hashes)
                ):
                    raise ValueError("invalid timestamp, lengths, or hash_ids")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_num}: {exc}") from exc
            previous = timestamp
            if input_len + output_len > max_model_len:
                if not drop_oversize:
                    raise ValueError(
                        f"{path}:{line_num}: input + output exceeds {max_model_len}; "
                        "choose a supported context or pass --drop-oversize"
                    )
                dropped += 1
                continue
            if origin is None:
                origin = timestamp
            rows.append({
                "timestamp": timestamp - origin,
                "input_length": input_len,
                "output_length": output_len,
                "hash_ids": hashes,
            })
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("selected workload has no requests")
    return rows, dropped


def percentile(values, level):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[math.ceil(level * len(ordered)) - 1], 3)


def summarize(result, rows, ttft_ms, tpot_ms):
    columns = ("input_lens", "output_lens", "ttfts", "itls", "errors", "start_times")
    if any(not isinstance(result.get(key), list) for key in columns):
        raise ValueError("missing detailed vLLM results; check --save-detailed")
    if any(len(result[key]) != len(rows) for key in columns):
        raise ValueError("result rows do not match the offered trace; check the CLI")
    duration = result["duration"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid benchmark duration")

    groups = {
        "all": [], "short": [], "long_input": [], "long_output": [], "both": [],
    }
    prompt_length_mismatches = 0
    for index, row in enumerate(rows):
        actual_output = result["output_lens"][index]
        success = not result["errors"][index] and actual_output > 0
        first = result["ttfts"][index] * 1000 if success else None
        tpot = None
        if success and actual_output == 1:
            tpot = 0.0
        elif success and result["itls"][index]:
            tpot = sum(result["itls"][index]) * 1000 / (actual_output - 1)
        qualified = (
            success and tpot is not None and first <= ttft_ms and tpot <= tpot_ms
        )
        long_input = row["input_length"] >= 8192
        long_output = row["output_length"] >= 1024
        group = "short"
        if long_input and long_output:
            group = "both"
        elif long_input:
            group = "long_input"
        elif long_output:
            group = "long_output"
        outcome = (success, qualified, first, tpot)
        groups[group].append(outcome)
        groups["all"].append(outcome)
        if success and result["input_lens"][index] != row["input_length"]:
            prompt_length_mismatches += 1

    summaries = {}
    for name, outcomes in groups.items():
        offered = len(outcomes)
        if not offered:
            continue
        success = sum(value[0] for value in outcomes)
        good = sum(value[1] for value in outcomes)
        firsts = [value[2] for value in outcomes if value[2] is not None]
        tpots = [value[3] for value in outcomes if value[3] is not None]
        summaries[name] = {
            "offered": offered,
            "completed_with_tokens": success,
            "failed_or_empty": offered - success,
            "qualified": good,
            "slo_attainment": round(good / offered, 6),
            "goodput_requests_per_s_including_drain": round(good / duration, 6),
            "ttft_p95_ms": percentile(firsts, 0.95),
            "ttft_p99_ms": percentile(firsts, 0.99),
            "tpot_p95_ms": percentile(tpots, 0.95),
            "tpot_p99_ms": percentile(tpots, 0.99),
            "tpot_observations": len(tpots),
        }
    return {
        "duration_seconds_including_drain": duration,
        "slo_ms": {"ttft": ttft_ms, "tpot": tpot_ms},
        "benchmark_prompt_length_mismatches": prompt_length_mismatches,
        "groups": summaries,
    }


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("steady", "prefill", "decode"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--requests", type=nonnegative_int, default=300)
    parser.add_argument("--rate", type=positive_float, required=True)
    args = parser.parse_args()
    if not args.requests:
        parser.error("--requests must be positive")
    write_trace(args.output, synthetic_trace(args.scenario, args.requests, args.rate))


if __name__ == "__main__":
    main()
