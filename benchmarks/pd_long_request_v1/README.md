# vLLM 长请求调度 V1

用两张 NVIDIA GPU 在单节点回放 1P1D 流量，分别验证 Prefill 的注意力计算预算和 Decode 的自适应 KV 准入。两项策略默认关闭，可独立启用。启动器使用本分支的 NIXL 代理和 `vllm bench serve`。

## 在 GPU 容器内跑通

```bash
git clone --branch pd-long-request-v1 https://github.com/yestring/vllm.git
cd vllm

# 已有能运行当前分支的 vLLM CUDA/NIXL 环境时，直接复用该环境的 Python。
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install nixl

PY="$PWD/.venv/bin/python"
BENCH="$PWD/benchmarks/pd_long_request_v1"
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/vllm-pd-v1.XXXXXX")
echo "$RUN_ROOT"

"$PY" "$BENCH/workload.py" steady "$RUN_ROOT/smoke.jsonl" --rate 1 --requests 8
"$PY" "$BENCH/run_baseline.py" --repo "$PWD" \
  --trace "$RUN_ROOT/smoke.jsonl" --output "$RUN_ROOT/smoke" \
  --model Qwen/Qwen3-0.6B --max-model-len 4096 \
  --prefill-gpu 0 --decode-gpu 1 --mode pull --eager \
  --ttft-ms 5000 --tpot-ms 500 --benchmark-timeout 120
```

设备参数可以传 GPU UUID。安装 NIXL 时按当前驱动选择兼容版本，见 [NIXL 安装说明](../../docs/features/nixl_connector_usage.md)。为对照保留 GPU 型号、拓扑、NIXL 版本和模型快照。`--dry-run` 只准备 trace 和启动清单，可在无 GPU 环境下验证参数传递。正式运行使用新输出目录。

## 四组对照

下面命令先用 `prefill` 合成负载。默认模型 `Qwen/Qwen3-8B`，上下文上限 32768。按基线容量调整 `RATE`，TTFT 和 TPOT 阈值在四组运行前确定。再用 `decode` 对照和 `steady` 回归负载重复这组运行。

```bash
RATE=${RATE:-2}
TTFT_MS=${TTFT_MS:-2500}
TPOT_MS=${TPOT_MS:-100}
SCENARIO=${SCENARIO:-prefill}
"$PY" "$BENCH/workload.py" "$SCENARIO" "$RUN_ROOT/$SCENARIO.jsonl" \
  --rate "$RATE" --requests 300

run_case() {
  name=$1
  shift
  "$PY" "$BENCH/run_baseline.py" --repo "$PWD" \
    --trace "$RUN_ROOT/$SCENARIO.jsonl" --output "$RUN_ROOT/$SCENARIO-$name" \
    --mode pull --prefill-gpu 0 --decode-gpu 1 \
    --ttft-ms "$TTFT_MS" --tpot-ms "$TPOT_MS" "$@"
}

PREFILL=(--prefill-arg=--prefill-attention-budget=8388608)
DECODE=(--decode-arg=--kv-admission-reserve-window-steps=32
        --decode-arg=--kv-admission-reserve-quantile=0.9)

run_case base
run_case prefill-only "${PREFILL[@]}"
run_case decode-only "${DECODE[@]}"
run_case both "${PREFILL[@]}" "${DECODE[@]}"
```

- **Prefill 预算**以每迭代的因果 query-key 配对数为单位。已有上下文为 `L`、新增 chunk 为 `c` 时，估算量为 `cL + c(c+1)/2`。`8,388,608` 是校准起点，单条请求从空上下文开始时约对应 4K。队列里有新请求时，初版把预算均分给进行中和等待中的请求。首个 chunk 允许最低一个 token 的进度。该开关适用于启用 chunked prefill 的专用 KV Producer、全因果注意力文本模型。观察队列、分组 TTFT 与吞吐后调整。
- **Decode 准入**使用最近调度迭代的活跃请求 KV 净增块量，估计下一轮需要的预留。`32` 步、`0.9` 分位是校准起点，已有固定 watermark 是预留量的下限。四组先用默认静态水位，再单独做调优后的固定水位对照。过高的固定水位会盖住动态值。需要诊断时可以用 `VLLM_LOGGING_LEVEL=DEBUG` 查看 D 日志的 `Decode KV growth`，性能对照使用默认日志级别。
- 四组保持同一模型和 Pull 传输路径。`--mode push` 可单独测本分支的原生 Push，作为通信对照。
- `steady` 使用短输入短输出。`prefill` 在中段插入 8K、16K 和 28K 冷输入，`decode` 在中段插入 2K 输出。三种负载具有相同的开放环到达时间。也可以用 [Mooncake FAST’25 公开 trace](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md) 替换输入。`--start N` 跳过前 N 行，`--limit N` 选取 N 条，`--time-scale 2` 把发送间隔拉长一倍。使用 `--drop-oversize` 时，超出上下文上限的请求数写入清单。

## 看结果

- `manifest.json` 保存 commit、设备参数、trace SHA-256、SLO 和完整命令。`summary.json` 保存总体和分组的 TTFT、TPOT、SLO 达标率及达标请求每秒。达标率分母包含失败请求，goodput 的时间分母包含清空队列的时间。
- `metrics.jsonl` 保存 P、D 分侧的队列、KV 占用、抢占、前缀命中以及 NIXL 指标。`benchmark.json` 保存原始逐请求结果，包含生成文本。启动失败先看 `prefill.log`、`decode.log`、`proxy.log` 或 `benchmark.log`。
- 比较 TTFT 和平均 TPOT 时同时看各组样本数与达标率。Mooncake 的匿名前缀哈希会重建输入，正式验收前核对 prompt 长度与前缀命中。NIXL 的传输 histogram 用于观察单次操作，整条请求的交接延迟需要请求级埋点。

首轮回传短流量和长请求突发各组的 `manifest.json`、`summary.json` 和 `metrics.jsonl`。判断 Prefill 优化时关注短请求 TTFT 和 P 队列，判断 Decode 优化时关注 D 抢占、KV 占用和长输出达标率。
