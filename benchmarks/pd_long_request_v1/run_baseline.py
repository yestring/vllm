"""Run a single-node 1P1D baseline using the selected vLLM checkout."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from workload import (
    nonnegative_int, positive_float, prepare_trace, summarize, write_trace,
)


HOST = "127.0.0.1"
PULL_PROXY = "tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
PUSH_PROXY = (
    "examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py"
)
METRIC_PREFIXES = tuple("vllm:" + name for name in (
    "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
    "num_preemptions", "prefix_cache_queries", "prefix_cache_hits",
    "external_prefix_cache_queries", "external_prefix_cache_hits",
    "request_queue_time_seconds", "request_prefill_time_seconds",
    "request_prefill_kv_computed_tokens", "nixl_",
))
DIRECT_HTTP = build_opener(ProxyHandler({}))


def url(port, route):
    return f"http://{HOST}:{port}{route}"


def check_ports(ports):
    if len(set(ports)) != len(ports):
        raise ValueError("all five HTTP and NIXL ports must be different")
    for port in ports:
        if not 0 < port < 65536:
            raise ValueError(f"invalid TCP port: {port}")
        with socket.socket() as listener:
            try:
                listener.bind((HOST, port))
            except OSError as exc:
                raise ValueError(f"port {port} is unavailable on {HOST}: {exc}") from exc


def wait_ready(endpoint, process, timeout, name):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} exited with {process.returncode}; see {name}.log"
            )
        try:
            with DIRECT_HTTP.open(endpoint, timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise RuntimeError(
        f"{name} did not become ready at {endpoint}; see {name}.log"
    )


def stop_process_group(process):
    # Each child has its own process group, including its engine workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def scrape_metrics(path, ports, interval, stop):
    def sample(stream):
        for role, port in ports.items():
            record = {"timestamp_unix": time.time(), "role": role}
            try:
                with DIRECT_HTTP.open(url(port, "/metrics"), timeout=2) as response:
                    lines = response.read().decode().splitlines()
                record["metrics"] = [
                    line for line in lines if line.startswith(METRIC_PREFIXES)
                ]
            except (URLError, TimeoutError, OSError) as exc:
                record["error"] = str(exc)
            stream.write(json.dumps(record) + "\n")
        stream.flush()

    with path.open("w", encoding="utf-8") as stream:
        sample(stream)
        while not stop.wait(interval):
            sample(stream)
        sample(stream)


def revision(repo):
    if not (repo / ".git").exists():
        return {"commit": None, "dirty": None}
    kwargs = {"cwd": repo, "capture_output": True, "text": True, "check": True}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], **kwargs).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], **kwargs).stdout
    return {"commit": commit, "dirty": bool(status)}


def command_plan(args, trace_file, count):
    cli = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    prefill_id = "pd-baseline-prefill"
    connector = "NixlConnector" if args.mode == "pull" else "NixlPushConnector"
    plan = {}
    for role, gpu, port, side_port, extra in (
        ("prefill", args.prefill_gpu, args.prefill_port, args.prefill_side_port,
         args.prefill_arg),
        ("decode", args.decode_gpu, args.decode_port, args.decode_side_port,
         args.decode_arg),
    ):
        config = {
            "kv_connector": connector,
            "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
            "engine_id": prefill_id if role == "prefill" else "pd-baseline-decode",
            "kv_load_failure_policy": "fail",
        }
        command = cli + [
            "serve", args.model, "--host", HOST, "--port", str(port),
            "--tensor-parallel-size", "1", "--max-model-len", str(args.max_model_len),
            "--dtype", args.dtype,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--kv-transfer-config", json.dumps(config),
        ]
        if args.eager:
            command.append("--enforce-eager")
        server_env = {
            "CUDA_VISIBLE_DEVICES": gpu,
            "VLLM_NIXL_SIDE_CHANNEL_HOST": HOST,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_port),
        }
        plan[role] = (command + extra, server_env)

    if args.mode == "pull":
        proxy = [
            PULL_PROXY, "--host", HOST,
            "--prefiller-hosts", HOST, "--prefiller-ports", str(args.prefill_port),
            "--decoder-hosts", HOST, "--decoder-ports", str(args.decode_port),
        ]
    else:
        proxy = [
            PUSH_PROXY, "--model", args.model,
            "--prefill", f"{HOST}:{args.prefill_port}",
            "--decode", f"{HOST}:{args.decode_port}",
            "--prefill-engine-id", prefill_id, "--prefill-kv-host", HOST,
            "--prefill-side-channel-port", str(args.prefill_side_port),
        ]
    plan["proxy"] = ([sys.executable, *proxy, "--port", str(args.proxy_port)], {})
    plan["benchmark"] = (cli + [
        "bench", "serve", "--backend", "openai", "--base-url", url(args.proxy_port, ""),
        "--endpoint", "/v1/completions", "--model", args.model,
        "--dataset-name", "timed_trace", "--dataset-path", str(trace_file),
        "--timed-trace-chunk-hash-size", "512",
        "--timed-trace-sec-multiplier", str(0.001 * args.time_scale),
        "--self-timed", "--num-prompts", str(count), "--seed", str(args.seed),
        "--ready-check-timeout-sec", "0", "--num-warmups", "0",
        "--save-result", "--save-detailed", "--result-dir", str(args.output),
        "--result-filename", "benchmark.json",
    ], {"PYTHONHASHSEED": str(args.seed)})
    return plan


def execute(args, plan):
    check_ports((args.prefill_port, args.decode_port, args.proxy_port,
                 args.prefill_side_port, args.decode_side_port))
    with ExitStack() as stack:
        def launch(name):
            command, env = plan[name]
            log_file = stack.enter_context((args.output / f"{name}.log").open("w"))
            print(f"Starting {name}; log: {log_file.name}", flush=True)
            process = subprocess.Popen(
                command, cwd=args.repo, env={**os.environ, **env}, stdout=log_file,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            stack.callback(stop_process_group, process)
            return process

        ports = {"prefill": args.prefill_port, "decode": args.decode_port}
        servers = {role: launch(role) for role in ports}
        for role, port in ports.items():
            wait_ready(url(port, "/health"), servers[role], args.startup_timeout, role)
        proxy = launch("proxy")
        proxy_health = "/healthcheck" if args.mode == "pull" else "/status"
        wait_ready(url(args.proxy_port, proxy_health), proxy, 30, "proxy")

        stop = threading.Event()
        metrics_thread = threading.Thread(
            target=scrape_metrics,
            args=(args.output / "metrics.jsonl", ports, args.scrape_interval, stop),
        )
        metrics_thread.start()
        try:
            benchmark = launch("benchmark")
            deadline = time.monotonic() + args.benchmark_timeout
            while benchmark.poll() is None:
                for name, process in (*servers.items(), ("proxy", proxy)):
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"{name} exited during replay; see {name}.log"
                        )
                if time.monotonic() >= deadline:
                    raise RuntimeError("benchmark exceeded --benchmark-timeout")
                time.sleep(0.25)
            if benchmark.returncode:
                raise RuntimeError(
                    f"benchmark exited with {benchmark.returncode}; see benchmark.log"
                )
        finally:
            stop.set()
            metrics_thread.join()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--mode", choices=("pull", "push"), default="pull")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=positive_float, default=0.9)
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--decode-port", type=int, default=8200)
    parser.add_argument("--proxy-port", type=int, default=8192)
    parser.add_argument("--prefill-side-port", type=int, default=5600)
    parser.add_argument("--decode-side-port", type=int, default=5601)
    parser.add_argument("--prefill-arg", action="append", default=[])
    parser.add_argument("--decode-arg", action="append", default=[])
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--ttft-ms", type=positive_float, required=True)
    parser.add_argument("--tpot-ms", type=positive_float, required=True)
    parser.add_argument("--start", type=nonnegative_int, default=0)
    parser.add_argument("--limit", type=nonnegative_int, default=0)
    parser.add_argument("--drop-oversize", action="store_true")
    parser.add_argument("--time-scale", type=positive_float, default=1.0)
    parser.add_argument("--scrape-interval", type=positive_float, default=1.0)
    parser.add_argument("--startup-timeout", type=positive_float, default=900)
    parser.add_argument("--benchmark-timeout", type=positive_float, default=3600)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.prefill_gpu == args.decode_gpu or
            any("," in gpu or not gpu for gpu in (args.prefill_gpu, args.decode_gpu))):
        parser.error("specify two distinct single GPU IDs")
    if args.max_model_len <= 0 or args.seed >= 2**32 or args.gpu_memory_utilization > 1:
        parser.error("require max-model-len > 0, seed < 2**32 and GPU utilization <= 1")
    return args


def main():
    args = parse_args()
    args.repo, args.output, args.trace = (path.resolve() for path in (
        args.repo, args.output, args.trace,
    ))
    proxy = PULL_PROXY if args.mode == "pull" else PUSH_PROXY
    try:
        if not (args.repo / proxy).is_file():
            raise ValueError(f"required proxy is missing: {args.repo / proxy}")
        rows, dropped = prepare_trace(
            args.trace, args.max_model_len, args.start, args.limit, args.drop_oversize,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        trace_file = args.output / "trace.jsonl"
        write_trace(trace_file, rows)
        plan = command_plan(args, trace_file, len(rows))
        manifest = {
            "mode": args.mode, "repo": str(args.repo), "revision": revision(args.repo),
            "source_trace": str(args.trace), "trace_start_row_offset": args.start,
            "trace_sha256": hashlib.sha256(trace_file.read_bytes()).hexdigest(),
            "offered": len(rows), "dropped_oversize": dropped, "dry_run": args.dry_run,
            "slo_ms": {"ttft": args.ttft_ms, "tpot": args.tpot_ms},
            "commands": plan,
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        )
        if args.dry_run:
            print(f"Prepared {len(rows)} requests and a manifest in {args.output}")
            return
        execute(args, plan)
        result = json.loads((args.output / "benchmark.json").read_text(encoding="utf-8"))
        report = summarize(result, rows, args.ttft_ms, args.tpot_ms)
        (args.output / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(report["groups"]["all"], indent=2))
        if not report["groups"]["all"]["completed_with_tokens"]:
            raise RuntimeError(
                "all requests failed or returned no tokens; see benchmark.log"
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"{exc}\nOutput directory: {args.output}") from exc


if __name__ == "__main__":
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    main()
