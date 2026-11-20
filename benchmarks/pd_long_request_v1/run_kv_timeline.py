#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run matched NIXL Pull traces and summarize the per-request KV timeline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_TRACE = Path("/workspace/vllm-benchmark.KNwXSi/matrix/cold.jsonl")
BENCHMARK_DIR = Path("benchmarks/pd_long_request_v1")
ARMS = (
    ("serial", 19100, (18_000,) * 6),
    ("overlap", 19200, (100, 500, 1_000) * 2),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/workspace/models/Qwen3-0.6B")
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="2")
    return parser.parse_args()


def select_requests(source):
    shorts, longs = [], []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["input_length"] == 1_024 and len(shorts) < 7:
                shorts.append(row)
            elif row["input_length"] == 28_672 and len(longs) < 6:
                longs.append(row)
            if len(shorts) == 7 and len(longs) == 6:
                return shorts, longs
    raise ValueError(f"{source} must contain at least seven 1K and six 28K inputs")


def write_arm_trace(path, shorts, longs, gaps):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(shorts[0], timestamp=0)) + "\n")
        for i, gap in enumerate(gaps):
            at = 25_000 + 35_000 * i
            stream.write(json.dumps(dict(longs[i], timestamp=at)) + "\n")
            stream.write(json.dumps(dict(shorts[i + 1], timestamp=at + gap)) + "\n")


def run_arm(args, runner, name, port, output):
    case = output / name
    timeline = case / "timeline.jsonl"
    command = [
        sys.executable, str(runner),
        "--repo", str(args.repo),
        "--model", args.model,
        "--max-model-len", "32768",
        "--mode", "pull",
        "--prefill-gpu", args.prefill_gpu,
        "--decode-gpu", args.decode_gpu,
        "--eager", "--seed", "0",
        "--prefill-arg=--enable-prefix-caching",
        "--decode-arg=--enable-prefix-caching",
        "--ttft-ms", "2500", "--tpot-ms", "100",
        "--scrape-interval", "0.25",
        "--prefill-port", str(port),
        "--decode-port", str(port + 1),
        "--proxy-port", str(port + 2),
        "--prefill-side-port", str(port + 3),
        "--decode-side-port", str(port + 4),
        "--benchmark-timeout", "420",
        "--startup-timeout", "180",
        "--trace", str(output / f"{name}.jsonl"),
        "--output", str(case),
    ]
    env = {**os.environ, "PD_KV_TIMELINE_PATH": str(timeline)}
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((str(args.repo), pythonpath)) if pythonpath else str(args.repo)
    )
    print(f"Running {name}: {case}", flush=True)
    try:
        subprocess.run(command, cwd=args.repo, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"{name} failed (exit {exc.returncode}); inspect {case}"
        ) from exc


def main():
    args = parse_args()
    args.repo = args.repo.resolve()
    output = args.output.resolve()
    runner = args.repo / BENCHMARK_DIR / "run_baseline.py"
    analyzer = Path(__file__).resolve().with_name("summarize_kv_timeline.py")
    try:
        for script in (runner, analyzer):
            if not script.is_file():
                raise ValueError(f"required script is missing: {script}")
        shorts, longs = select_requests(args.trace.resolve())
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError(f"output directory must be empty: {output}")
        print(f"Results directory: {output}", flush=True)
        for name, _, gaps in ARMS:
            write_arm_trace(output / f"{name}.jsonl", shorts, longs, gaps)

        for name, port, _ in ARMS:
            run_arm(args, runner, name, port, output)

        cases = [str(output / name) for name, _, _ in ARMS]
        report = output / "timeline-report.txt"
        with report.open("x", encoding="utf-8") as stream:
            subprocess.run(
                [sys.executable, str(analyzer), *cases], stdout=stream, check=True
            )
        print(report.read_text(encoding="utf-8"), end="")
        print(f"Timeline report: {report}")
    except KeyboardInterrupt:
        print(f"Interrupted. Partial results: {output}", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"{exc}\nResults directory: {output}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
