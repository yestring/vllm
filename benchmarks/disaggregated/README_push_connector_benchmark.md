# NIXL Push Connector benchmark

This benchmark compares whole-request and layerwise NIXL push transfer through
the OpenAI-compatible disaggregated proxy. It records per-request measurements,
aggregated percentiles, throughput, failures, and presentation-ready plots.
It also snapshots P's `/metrics` endpoint to report NIXL transfer count, bytes,
transfer time, post time, and effective transfer bandwidth.

Run the benchmark in the same Python environment used to launch vLLM. If the
local `.venv` does not already contain the optional plotting dependencies:

```bash
uv pip install --python .venv/bin/python aiohttp transformers matplotlib
```

## 1. Run the baseline

Start P on GPU 0, D on GPU 1, and the proxy on port 8000 with layerwise
transfer disabled. Then run:

```bash
.venv/bin/python benchmarks/disaggregated/push_connector_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --model /home/hx/Qwen2.5-1.5B-Instruct \
  --label baseline \
  --output-dir benchmark_results/push_baseline \
  --input-tokens 128,512,1024,2048,3072 \
  --concurrency 1,2,4,8,16 \
  --output-tokens 128 \
  --max-model-len 4096 \
  --requests 30
```

## 2. Run layerwise transfer

Restart P and D with the same configuration and
`kv_connector_extra_config.enable_layerwise_transfer=true`. Keep the model,
ports, block size, memory utilization, CUDA graph settings, and all other
arguments unchanged.

```bash
.venv/bin/python benchmarks/disaggregated/push_connector_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --model /home/hx/Qwen2.5-1.5B-Instruct \
  --label layerwise \
  --output-dir benchmark_results/push_layerwise \
  --input-tokens 128,512,1024,2048,3072 \
  --concurrency 1,2,4,8,16 \
  --output-tokens 128 \
  --max-model-len 4096 \
  --requests 30
```

Use at least 100 requests per scenario for final reported numbers. Thirty is a
quicker functional sweep.

## 3. Generate plots

```bash
.venv/bin/python benchmarks/disaggregated/plot_push_connector_results.py \
  benchmark_results/push_baseline \
  benchmark_results/push_layerwise \
  --output-dir benchmark_results/push_comparison
```

Each run produces:

- `requests.csv`: one row per request, including errors;
- `summary.csv`: p50/p90/p99 TTFT, TPOT, E2E, ITL and throughput;
- `metadata.json`: arguments, proxy topology and metric definitions.

The plot command produces PNG and PDF figures, `comparison.csv`, and a TTFT
improvement heatmap.

## Measurement notes

- TTFT is measured at the client, from HTTP request start to the first
  non-empty streamed token.
- Every scenario must satisfy `input_tokens + output_tokens <= max_model_len`.
  Set `--max-model-len` to exactly the value used to launch P and D.
- TPOT is `(E2E - TTFT) / (output tokens - 1)`.
- Prompts are deterministic random token sequences to minimize accidental
  cross-request prefix-cache hits.
- The benchmark is closed-loop: at most `--concurrency` requests are in flight.
- Run baseline and layerwise configurations alternately at least three times
  if the expected difference is small.
- Monitor GPU clocks and utilization separately with `nvidia-smi dmon`; client
  measurements cannot directly determine compute/communication overlap.
- The default P metrics endpoint is `http://127.0.0.1:8100/metrics`. Override it
  with `--prefill-metrics-url`, or pass an empty string to disable scraping.
