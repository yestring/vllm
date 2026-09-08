# SPDX-License-Identifier: Apache-2.0
"""Benchmark an OpenAI-compatible disaggregated push proxy."""

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
from transformers import AutoTokenizer


@dataclass
class RequestResult:
    label: str
    input_tokens: int
    output_tokens_requested: int
    concurrency: int
    request_index: int
    success: bool
    actual_input_tokens: int
    actual_output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    mean_itl_ms: float
    error: str


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class PromptFactory:
    """Create deterministic prompts with little cross-request prefix sharing."""

    def __init__(self, tokenizer: Any, seed: int):
        self.tokenizer = tokenizer
        self.seed = seed
        special_ids = set(tokenizer.all_special_ids)
        vocab_size = len(tokenizer)
        self.candidate_ids = [
            token_id
            for token_id in range(vocab_size)
            if token_id not in special_ids
        ]

    def make(self, target_tokens: int, request_index: int) -> tuple[str, int]:
        rng = random.Random(self.seed + request_index * 1_000_003 + target_tokens)
        token_ids = rng.choices(self.candidate_ids, k=target_tokens)
        prompt = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        actual = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if actual > target_tokens:
            normalized_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )[:target_tokens]
            prompt = self.tokenizer.decode(
                normalized_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            actual = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        return prompt, actual


async def run_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    model: str,
    prompt: str,
    actual_input_tokens: int,
    output_tokens: int,
    label: str,
    input_tokens: int,
    concurrency: int,
    request_index: int,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = 0.0
    first_token_at: float | None = None
    token_event_times: list[float] = []
    generated_text = ""
    error = ""
    response_preview = bytearray()

    try:
        async with semaphore:
            started = time.perf_counter()
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {body[:500]}")

                buffer = b""
                async for chunk in response.content.iter_any():
                    if len(response_preview) < 500:
                        response_preview.extend(
                            chunk[: 500 - len(response_preview)]
                        )
                    buffer += chunk
                    while b"\n" in buffer:
                        raw_line, buffer = buffer.split(b"\n", 1)
                        line = raw_line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == b"[DONE]":
                            continue
                        event = json.loads(data)
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        text = choices[0].get("text", "")
                        if not text:
                            continue
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        token_event_times.append(now)
                        generated_text += text
                finished = time.perf_counter()
                if first_token_at is None:
                    preview = response_preview.decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                    error = (
                        "stream ended without a generated token"
                        + (f": {preview}" if preview else "")
                    )
    except Exception as exc:
        finished = time.perf_counter()
        error = f"{type(exc).__name__}: {exc}"

    successful = first_token_at is not None and not error
    actual_output_tokens = 0
    if generated_text:
        actual_output_tokens = len(
            TOKENIZER.encode(generated_text, add_special_tokens=False)
        )

    e2e_ms = (finished - started) * 1000 if started else math.nan
    ttft_ms = (
        (first_token_at - started) * 1000
        if first_token_at is not None
        else math.nan
    )
    tpot_ms = (
        (finished - first_token_at) * 1000 / (actual_output_tokens - 1)
        if first_token_at is not None and actual_output_tokens > 1
        else math.nan
    )
    itls = [
        (current - previous) * 1000
        for previous, current in zip(token_event_times, token_event_times[1:])
    ]
    mean_itl_ms = statistics.fmean(itls) if itls else math.nan

    return RequestResult(
        label=label,
        input_tokens=input_tokens,
        output_tokens_requested=output_tokens,
        concurrency=concurrency,
        request_index=request_index,
        success=successful,
        actual_input_tokens=actual_input_tokens,
        actual_output_tokens=actual_output_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        mean_itl_ms=mean_itl_ms,
        error=error,
    )


def summarize(
    results: list[RequestResult],
    duration_s: float,
) -> dict[str, Any]:
    successful = [result for result in results if result.success]

    def metric(name: str) -> list[float]:
        return [
            value
            for result in successful
            if math.isfinite(value := float(getattr(result, name)))
        ]

    row: dict[str, Any] = {
        "label": results[0].label,
        "input_tokens": results[0].input_tokens,
        "output_tokens_requested": results[0].output_tokens_requested,
        "concurrency": results[0].concurrency,
        "requests": len(results),
        "successful_requests": len(successful),
        "failure_rate": 1 - len(successful) / len(results),
        "duration_s": duration_s,
        "request_throughput_rps": len(successful) / duration_s,
        "output_throughput_tps": (
            sum(result.actual_output_tokens for result in successful) / duration_s
        ),
        "mean_actual_input_tokens": (
            statistics.fmean(result.actual_input_tokens for result in successful)
            if successful
            else math.nan
        ),
    }
    for name in ("ttft_ms", "tpot_ms", "e2e_ms", "mean_itl_ms"):
        values = metric(name)
        row[f"{name}_mean"] = statistics.fmean(values) if values else math.nan
        row[f"{name}_p50"] = percentile(values, 0.50)
        row[f"{name}_p90"] = percentile(values, 0.90)
        row[f"{name}_p99"] = percentile(values, 0.99)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_prometheus_sums(text: str) -> dict[str, float]:
    wanted = {
        "vllm:nixl_xfer_time_seconds_sum",
        "vllm:nixl_xfer_time_seconds_count",
        "vllm:nixl_bytes_transferred_sum",
        "vllm:nixl_bytes_transferred_count",
        "vllm:nixl_post_time_seconds_sum",
    }
    totals = {name: 0.0 for name in wanted}
    found: set[str] = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, _, raw_value = line.rpartition(" ")
        metric_name = sample.split("{", 1)[0]
        if metric_name not in wanted:
            continue
        try:
            totals[metric_name] += float(raw_value)
        except ValueError:
            continue
        found.add(metric_name)
    return {name: value for name, value in totals.items() if name in found}


async def scrape_metrics(
    session: aiohttp.ClientSession,
    url: str | None,
) -> dict[str, float]:
    if not url:
        return {}
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return {}
            return parse_prometheus_sums(await response.text())
    except aiohttp.ClientError:
        return {}


def add_nixl_metrics(
    summary: dict[str, Any],
    before: dict[str, float],
    after: dict[str, float],
) -> None:
    def delta(name: str) -> float:
        if name not in before or name not in after:
            return math.nan
        return max(0.0, after[name] - before[name])

    xfer_seconds = delta("vllm:nixl_xfer_time_seconds_sum")
    transferred_bytes = delta("vllm:nixl_bytes_transferred_sum")
    summary["nixl_transfer_count"] = delta(
        "vllm:nixl_xfer_time_seconds_count"
    )
    summary["nixl_transfer_time_s"] = xfer_seconds
    summary["nixl_post_time_s"] = delta("vllm:nixl_post_time_seconds_sum")
    summary["nixl_transferred_mib"] = transferred_bytes / 2**20
    summary["nixl_effective_gib_per_s"] = (
        transferred_bytes / 2**30 / xfer_seconds
        if math.isfinite(xfer_seconds) and xfer_seconds > 0
        else math.nan
    )


async def warmup(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    factory: PromptFactory,
    input_tokens: int,
    output_tokens: int,
    count: int,
) -> None:
    semaphore = asyncio.Semaphore(1)
    for index in range(count):
        prompt, actual = factory.make(input_tokens, -index - 1)
        result = await run_request(
            session,
            semaphore,
            url,
            model,
            prompt,
            actual,
            output_tokens,
            "warmup",
            input_tokens,
            1,
            -index - 1,
        )
        if not result.success:
            raise RuntimeError(f"warmup request failed: {result.error}")


async def benchmark(args: argparse.Namespace) -> None:
    oversized = [
        input_tokens
        for input_tokens in args.input_tokens
        if input_tokens + args.output_tokens > args.max_model_len
    ]
    if oversized:
        raise ValueError(
            "input_tokens + output_tokens exceeds --max-model-len "
            f"({args.max_model_len}) for input sizes {oversized}. "
            "Reduce --input-tokens/--output-tokens or pass the actual server "
            "--max-model-len value."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factory = PromptFactory(TOKENIZER, args.seed)
    raw_results: list[RequestResult] = []
    summaries: list[dict[str, Any]] = []
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:
        status_url = args.base_url.rstrip("/") + "/status"
        async with session.get(status_url) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"proxy readiness check failed with HTTP {response.status}"
                )
            proxy_status = await response.json()

        for input_tokens in args.input_tokens:
            await warmup(
                session,
                args.base_url.rstrip("/") + "/v1/completions",
                args.model,
                factory,
                input_tokens,
                args.output_tokens,
                args.warmup_requests,
            )
            for concurrency in args.concurrency:
                prompts = [
                    factory.make(
                        input_tokens,
                        input_tokens * 10_000_000
                        + concurrency * 100_000
                        + index,
                    )
                    for index in range(args.requests)
                ]
                semaphore = asyncio.Semaphore(concurrency)
                metrics_before = await scrape_metrics(
                    session,
                    args.prefill_metrics_url,
                )
                scenario_started = time.perf_counter()
                tasks = [
                    run_request(
                        session,
                        semaphore,
                        args.base_url.rstrip("/") + "/v1/completions",
                        args.model,
                        prompt,
                        actual,
                        args.output_tokens,
                        args.label,
                        input_tokens,
                        concurrency,
                        index,
                    )
                    for index, (prompt, actual) in enumerate(prompts)
                ]
                results = await asyncio.gather(*tasks)
                duration_s = time.perf_counter() - scenario_started
                if args.metrics_settle_seconds:
                    await asyncio.sleep(args.metrics_settle_seconds)
                metrics_after = await scrape_metrics(
                    session,
                    args.prefill_metrics_url,
                )
                raw_results.extend(results)
                summary = summarize(results, duration_s)
                add_nixl_metrics(summary, metrics_before, metrics_after)
                summaries.append(summary)
                # Checkpoint after every scenario so a later warmup or server
                # failure does not discard an otherwise valid long sweep.
                write_csv(
                    output_dir / "requests.csv",
                    [asdict(row) for row in raw_results],
                )
                write_csv(output_dir / "summary.csv", summaries)
                print(
                    f"[{args.label}] input={input_tokens} "
                    f"concurrency={concurrency}: "
                    f"TTFT p50={summary['ttft_ms_p50']:.2f} ms, "
                    f"TPOT p50={summary['tpot_ms_p50']:.2f} ms, "
                    f"throughput={summary['output_throughput_tps']:.2f} tok/s, "
                    f"NIXL={summary['nixl_effective_gib_per_s']:.2f} GiB/s, "
                    f"failures={summary['failure_rate']:.1%}",
                    flush=True,
                )

    write_csv(output_dir / "requests.csv", [asdict(row) for row in raw_results])
    write_csv(output_dir / "summary.csv", summaries)
    metadata = {
        "arguments": vars(args),
        "proxy_status": proxy_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metric_notes": {
            "ttft_ms": "client POST start to first non-empty SSE token",
            "tpot_ms": "(E2E - TTFT) / (tokenizer output tokens - 1)",
            "mean_itl_ms": "mean gap between non-empty SSE text events",
            "throughput": "successful output tokens / scenario wall time",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--input-tokens",
        type=parse_int_list,
        default=parse_int_list("128,512,1024,2048,3072"),
    )
    parser.add_argument(
        "--concurrency",
        type=parse_int_list,
        default=parse_int_list("1,2,4,8,16"),
    )
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--prefill-metrics-url",
        default="http://127.0.0.1:8100/metrics",
    )
    parser.add_argument("--metrics-settle-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


TOKENIZER: Any


def main() -> None:
    global TOKENIZER
    args = build_parser().parse_args()
    tokenizer_name = args.tokenizer or args.model
    TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
