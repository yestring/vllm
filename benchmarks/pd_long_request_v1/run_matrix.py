"""Run all four P/D policy combinations on paired cold/warm prefix traces."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from workload import nonnegative_int, positive_float, synthetic_trace, write_trace


def paired_traces(rows, prefix_tokens):
    """Keep the prompts identical and isolate only cold cache keys."""
    if not prefix_tokens or prefix_tokens % 512:
        raise ValueError("--prefix-tokens must be a positive multiple of 512")
    prefix_blocks = prefix_tokens // 512
    # Synthetic workload hash IDs are nonnegative. Give the shared prefix
    # its own ID range, leaving every request's suffix distinct.
    shared_hashes = list(range(-prefix_blocks, 0))
    cold, warm, eligible = [], [], 0
    for index, row in enumerate(rows):
        use_prefix = row["input_length"] > prefix_tokens
        hashes = row["hash_ids"]
        if use_prefix:
            hashes = shared_hashes + hashes[prefix_blocks:]
            eligible += 1
        original = {**row, "hash_ids": hashes, "cache_salt": f"pd-prefix-{index}"}
        cold.append(original)
        warm.append(
            {**original, "cache_salt": "pd-prefix-shared"} if use_prefix else original
        )
    if eligible < 2:
        raise ValueError(
            "At least two requests must have more tokens than --prefix-tokens"
        )
    return cold, warm, eligible


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=("steady", "prefill", "decode"), default="prefill",
    )
    parser.add_argument("--rate", type=positive_float, default=2)
    parser.add_argument("--requests", type=nonnegative_int, default=300)
    parser.add_argument("--prefix-tokens", type=nonnegative_int, default=4096)
    parser.add_argument("--prefill-budget", type=nonnegative_int, default=8388608)
    parser.add_argument("--decode-window", type=nonnegative_int, default=32)
    parser.add_argument("--decode-quantile", type=positive_float, default=0.9)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--attention-backend", help="Pin the same backend in P and D")
    parser.add_argument("--max-model-len", type=nonnegative_int, default=32768)
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--benchmark-timeout", type=positive_float, default=3600)
    parser.add_argument("--ttft-ms", type=positive_float, required=True)
    parser.add_argument("--tpot-ms", type=positive_float, required=True)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.prefill_budget or not args.decode_window or args.decode_quantile > 1:
        parser.error(
            "require positive prefill budget, decode window, and quantile <= 1"
        )
    return args


def run_matrix(args):
    output = args.output.resolve()
    cold, warm, eligible = paired_traces(
        synthetic_trace(args.scenario, args.requests, args.rate), args.prefix_tokens,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_trace(output / "cold.jsonl", cold)
    write_trace(output / "warm.jsonl", warm)

    matrix = {
        "scenario": args.scenario,
        "rate": args.rate,
        "requests_per_run": args.requests,
        "shared_prefix_tokens": args.prefix_tokens,
        "shared_prefix_eligible_requests": eligible,
        "dry_run": args.dry_run,
        "runs": [],
    }
    baseline = Path(__file__).with_name("run_baseline.py").resolve()
    common = [
        sys.executable, str(baseline), "--repo", str(args.repo.resolve()),
        "--mode", "pull", "--model", args.model,
        "--max-model-len", str(args.max_model_len),
        "--prefill-gpu", args.prefill_gpu, "--decode-gpu", args.decode_gpu,
        "--ttft-ms", str(args.ttft_ms), "--tpot-ms", str(args.tpot_ms),
        "--benchmark-timeout", str(args.benchmark_timeout), "--seed", str(args.seed),
        "--prefill-arg=--enable-prefix-caching", "--decode-arg=--enable-prefix-caching",
    ]
    if args.dry_run:
        common.append("--dry-run")
    if args.eager:
        common.append("--eager")
    if args.attention_backend:
        for role in ("prefill", "decode"):
            common.append(f"--{role}-arg=--attention-backend={args.attention_backend}")
    prefill = [f"--prefill-arg=--prefill-attention-budget={args.prefill_budget}"]
    decode = [
        f"--decode-arg=--kv-admission-reserve-window-steps={args.decode_window}",
        f"--decode-arg=--kv-admission-reserve-quantile={args.decode_quantile}",
    ]
    matrix_path = output / "matrix.json"
    policies = (
        ("base", []), ("prefill-only", prefill), ("decode-only", decode),
        ("both", prefill + decode),
    )
    for index, (name, policy) in enumerate(policies):
        # Alternate each pair's order to reduce systematic warm-after-cold bias.
        phases = ("cold", "warm") if index % 2 == 0 else ("warm", "cold")
        for phase in phases:
            run = output / f"{name}-{phase}"
            print(f"Running {run.name}", flush=True)
            command = common + policy + [
                "--trace", str(output / f"{phase}.jsonl"), "--output", str(run),
            ]
            result = subprocess.run(command, check=False)
            row = {"policy": name, "prefix": phase, "directory": run.name,
                   "exit_code": result.returncode}
            if result.returncode == 0 and not args.dry_run:
                summary = json.loads(
                    (run / "summary.json").read_text(encoding="utf-8")
                )
                row["prefill_prefix_cache"] = summary["prefix_cache"]["prefill"]
                row["groups"] = summary["groups"]
            matrix["runs"].append(row)
            matrix_path.write_text(
                json.dumps(matrix, indent=2) + "\n", encoding="utf-8",
            )
            result.check_returncode()
    if not args.dry_run:
        by_case = {(row["policy"], row["prefix"]): row for row in matrix["runs"]}
        matrix["prefill_hit_rate_delta"] = {}
        for name, _ in policies:
            cold_rate = by_case[name, "cold"]["prefill_prefix_cache"]["hit_rate"]
            warm_rate = by_case[name, "warm"]["prefill_prefix_cache"]["hit_rate"]
            matrix["prefill_hit_rate_delta"][name] = (
                round(warm_rate - cold_rate, 6)
                if cold_rate is not None and warm_rate is not None
                else None
            )
        matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"Comparison: {matrix_path}")


if __name__ == "__main__":
    args = parse_args()
    try:
        run_matrix(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
