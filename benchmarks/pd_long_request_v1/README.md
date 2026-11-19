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

## 自动跑八组冷、热前缀对照

`run_matrix.py` 固定同一模型、GPU、Pull 路径、SLO 与请求轨迹。冷、热条件的 prompt 内容、到达时间和输入输出长度逐条相同，两组都启用 P、D 的前缀缓存。`cold` 使用逐请求独立 `cache_salt`，`warm` 给输入长度大于共享前缀长度的请求使用共享 salt。每场重新启动 P、D 引擎，`warm` 的前缀由回放中的首批请求填充。每种条件都跑 `base`、`prefill-only`、`decode-only`、`both`，共八场。

默认负载为 `prefill`，使用 `Qwen/Qwen3-8B`、32768 上下文上限和 4096 token 的共享前缀。下面数值可用于首轮校准。正式对照前确定发送速率与延迟目标。

```bash
"$PY" "$BENCH/run_matrix.py" --repo "$PWD" --output "$RUN_ROOT/matrix" \
  --prefill-gpu 0 --decode-gpu 1 --rate 2 --requests 300 \
  --ttft-ms 2500 --tpot-ms 100
```

仅检查启动配置时，指定新的 `--output` 目录，可加 `--dry-run --scenario steady --model Qwen/Qwen3-0.6B --max-model-len 4096 --prefix-tokens 512 --requests 8 --eager`，生成八份启动清单。做 Decode 压力对照时，用 `--scenario decode --prefix-tokens 512`。`matrix.json` 记录各场的入口与状态，每场细节见 `<policy>-<cold|warm>/`。

- **Prefill 预算**以每迭代的因果 query-key 配对数为单位。已有上下文为 `L`、新增 chunk 为 `c` 时，估算量为 `cL + c(c+1)/2`。`8,388,608` 是校准起点，单条请求从空上下文开始时约对应 4K。队列里有新请求时，初版把预算均分给进行中和等待中的请求。首个 chunk 允许最低一个 token 的进度。该开关适用于启用 chunked prefill 的专用 KV Producer、全因果注意力文本模型。观察队列、分组 TTFT 与吞吐后调整。
- **Decode 准入**使用最近调度迭代的活跃请求 KV 净增块量，估计下一轮需要的预留。`32` 步、`0.9` 分位是校准起点，已有固定 watermark 是预留量的下限。四组先用默认静态水位，再单独做调优后的固定水位对照。过高的固定水位会盖住动态值。需要诊断时可以用 `VLLM_LOGGING_LEVEL=DEBUG` 查看 D 日志的 `Decode KV growth`，性能对照使用默认日志级别。
- `steady` 使用短输入短输出，`prefill` 在中段插入 8K、16K 和 28K 输入，`decode` 在中段插入 2K 输出。三种负载具有相同的开放环到达时间。可以用 `run_baseline.py --mode push` 单独测原生 Push，或者用 [Mooncake FAST’25 公开 trace](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md) 回放。`--start N` 跳过前 N 行，`--limit N` 选取 N 条，`--time-scale 2` 把发送间隔拉长一倍。使用 `--drop-oversize` 时，超出上下文上限的请求数写入清单。

## 单独测量 Attention 成本

在一张 GPU 上运行独立微基准，采集不同已有上下文、chunk 和 batch 下的 CUDA 耗时，作为解释 Prefill 预算的证据。下例同时保存 JSON 和 CSV。模型 head 数或 KV 布局不同时，按目标配置修改参数。可先加 `--dry-run` 检查形状矩阵。

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" "$BENCH/profile_attention.py" \
  --context-lengths 0 4096 16384 28672 \
  --chunk-sizes 64 256 512 --batch-sizes 1 4 8 \
  --warmup 10 --repeats 40 --check \
  --output "$RUN_ROOT/attention.json" --csv "$RUN_ROOT/attention.csv"
```

脚本单独计时一层 paged FlashAttention，K/V 已经写入页，输出包括中位数和 P95。模型投影、KV 写入、多卡通信和请求排队另由端到端实验覆盖。若要对照同一后端，可在矩阵命令加 `--attention-backend FLASH_ATTN`，并核对启动日志、FlashAttention 版本和 KV 布局。用 Nsight Systems 时，可通过 `nsys profile --trace=cuda,nvtx` 包住命令，再加 `--nvtx` 记录形状区间。

## 看结果

- `manifest.json` 保存 commit、设备参数、trace SHA-256、SLO 和完整命令。`summary.json` 保存总体和分组的 TTFT、TPOT、SLO 达标率及达标请求每秒。达标率分母包含失败请求，goodput 的时间分母包含清空队列的时间。
- `metrics.jsonl` 保存 P、D 分侧的队列、KV 占用、抢占、前缀命中以及 NIXL 指标。`benchmark.json` 保存原始逐请求结果，包含生成文本。启动失败先看 `prefill.log`、`decode.log`、`proxy.log` 或 `benchmark.log`。
- 先查 `matrix.json` 的 `prefill_hit_rate_delta`，再查各场 `summary.json` 的 `prefix_cache`，核对 P 侧 `queries`、`hits` 的 token 数增量与 `hit_rate`。若热组与冷组的实际命中率没有拉开，先核对 prefix 长度、warm 请求间隔、prompt 重建后的 token 偏差和 cache 回收。先在同一前缀条件内比较策略，再比较冷、热环境下的收益差异。
- 比较 TTFT 和平均 TPOT 时同时看各组样本数与达标率。NIXL 的传输 histogram 用于观察单次操作，整条请求的交接延迟需要请求级埋点。

首轮回传短流量和长请求突发各组的 `manifest.json`、`summary.json` 和 `metrics.jsonl`。判断 Prefill 优化时关注短请求 TTFT 和 P 队列，判断 Decode 优化时关注 D 抢占、KV 占用和长输出达标率。
