#!/usr/bin/env python3
"""
vLLM KV Cache Eviction Benchmark (外置服务版)

运行前提：
- 已手动启动两个 vLLM 服务：
  - Loss-aware: 端口 8000 (需配置 --enable-loss-eviction)
  - LRU: 端口 8002

本脚本会依次向两个服务发送相同请求，对比性能指标。
"""
import time
import re
import json
import requests
import statistics
import sys
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# 1. 数据生成器
# ============================================================

class PrefixSharingDataset:
    """生成带有可控共享前缀的测试数据集（共享前缀至少 20 tokens）"""
    
    # 长共享前缀（固定，重复多次以确保长度）
    SHARED_PREFIX = (
        "You are a helpful assistant. Please provide a detailed answer to the following question. "
        "Ensure your response is accurate, clear, and concise."
    )  # 约 16 个单词，约 18~20 tokens（根据模型 tokenizer 略有差异）
    
    # 如果仍不够，可重复拼接
    SHARED_PREFIX_LONG = SHARED_PREFIX + " " + SHARED_PREFIX  # 约 36 tokens，绝对充足
    
    # 独特后缀（用于区分不同请求，但长度适中）
    UNIQUE_SUFFIXES = [
        "Explain the concept of attention mechanism.",
        "Write a Python function to compute fibonacci numbers.",
        "What are the advantages of using async/await in Python?",
        "Describe the process of training a neural network.",
        "How does garbage collection work in Java?",
        "Explain the difference between TCP and UDP.",
        "Write a simple REST API using Flask.",
        "What is the time complexity of quicksort?",
        "Describe the architecture of a transformer model.",
        "How to optimize database queries?",
    ]
    
    # 独有前缀（用于非共享请求）
    UNIQUE_PREFIXES = [
        "User query: Explain the concept of attention mechanism.",
        "User query: Write a Python function to compute fibonacci numbers.",
        "User query: What are the advantages of using async/await in Python?",
        "User query: Describe the process of training a neural network.",
        "User query: How does garbage collection work in Java?",
        "User query: Explain the difference between TCP and UDP.",
        "User query: Write a simple REST API using Flask.",
        "User query: What is the time complexity of quicksort?",
        "User query: Describe the architecture of a transformer model.",
        "User query: How to optimize database queries?",
    ]
    
    @classmethod
    def generate_dataset(
        cls,
        num_requests: int = 500,
        prefix_share_rate: float = 0.5,
        output_path: str = None
    ) -> List[str]:
        """
        生成 prompt 列表。
        - 共享请求：使用相同长前缀 + 随机后缀
        - 独有请求：使用不同前缀 + 随机后缀
        """
        # 确保共享前缀足够长（至少 16 tokens）
        # 这里直接用 SHARED_PREFIX_LONG（约 36 tokens）
        shared_prefix = cls.SHARED_PREFIX_LONG
        
        prompts = []
        num_shared = int(num_requests * prefix_share_rate)
        num_unique = num_requests - num_shared
        
        # 生成共享请求
        for _ in range(num_shared):
            suffix = random.choice(cls.UNIQUE_SUFFIXES)
            prompts.append(f"{shared_prefix} {suffix}")
        
        # 生成独有请求
        for _ in range(num_unique):
            prefix = random.choice(cls.UNIQUE_PREFIXES)
            suffix = random.choice(cls.UNIQUE_SUFFIXES)
            prompts.append(f"{prefix} {suffix}")
        
        # 打乱顺序
        random.shuffle(prompts)
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(prompts, f, indent=2)
            print(f"Dataset saved to {output_path} (size={len(prompts)})")
        
        return prompts


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class RequestMetrics:
    request_id: str
    prompt: str
    arrival_time: float
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    total_tokens: int = 0
    ttft: Optional[float] = None
    tpot: Optional[float] = None
    status: str = "pending"


@dataclass
class KVCacheTestResult:
    eviction_policy: str          # "lru" or "loss_aware"
    prefix_share_rate: float
    concurrency: int
    num_requests: int
    successful: int
    failed: int
    ttfts: List[float] = field(default_factory=list)
    tpots: List[float] = field(default_factory=list)
    throughput: float = 0.0
    prefix_hit_rate: float = 0.0
    kv_cache_usage: float = 0.0
    num_evictions: int = 0
    short_ttfts: List[float] = field(default_factory=list)
    long_ttfts: List[float] = field(default_factory=list)
    
    def compute_stats(self) -> Dict[str, Any]:
        stats = {}
        stats['success_rate'] = self.successful / (self.successful + self.failed) if (self.successful + self.failed) > 0 else 0
        stats['throughput'] = self.throughput
        if self.ttfts:
            stats['ttft_avg'] = statistics.mean(self.ttfts) * 1000
            stats['ttft_p50'] = statistics.median(self.ttfts) * 1000
            stats['ttft_p95'] = np.percentile(self.ttfts, 95) * 1000 if len(self.ttfts) >= 20 else max(self.ttfts) * 1000
            stats['ttft_p99'] = np.percentile(self.ttfts, 99) * 1000 if len(self.ttfts) >= 20 else max(self.ttfts) * 1000
            stats['ttft_std'] = statistics.stdev(self.ttfts) * 1000 if len(self.ttfts) > 1 else 0
        if self.tpots:
            stats['tpot_avg'] = statistics.mean(self.tpots) * 1000
        stats['prefix_hit_rate'] = self.prefix_hit_rate
        stats['kv_cache_usage'] = self.kv_cache_usage
        stats['num_evictions'] = self.num_evictions
        if self.short_ttfts:
            stats['ttft_short_avg'] = statistics.mean(self.short_ttfts) * 1000
        if self.long_ttfts:
            stats['ttft_long_avg'] = statistics.mean(self.long_ttfts) * 1000
        if self.short_ttfts and self.long_ttfts:
            stats['long_short_ttft_ratio'] = statistics.mean(self.long_ttfts) / statistics.mean(self.short_ttfts)
        return stats


# ============================================================
# 3. 测试引擎（只发请求，不管理服务）
# ============================================================

class KVCacheBenchmarkEngine:
    def __init__(
        self,
        model_path: str,
        port_loss_aware: int = 8000,
        port_lru: int = 8002,
        max_tokens_per_request: int = 50,
        timeout: int = 180
    ):
        self.model_path = model_path
        self.port_loss_aware = port_loss_aware
        self.port_lru = port_lru
        self.max_tokens_per_request = max_tokens_per_request
        self.timeout = timeout

    def _fetch_metrics(self, port: int) -> Dict[str, float]:
        metrics = {
            'hits': 0.0,
            'queries': 0.0,
            'kv_cache_usage': 0.0,
            'num_evictions': 0
        }
        try:
            resp = requests.get(f"http://localhost:{port}/metrics", timeout=5)
            if resp.status_code != 200:
                return metrics
    
            for line in resp.text.splitlines():
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                full_name = parts[0]          # e.g. vllm:prefix_cache_hits_total{...}
                value = float(parts[1])
    
                # 提取纯指标名
                if '{' in full_name:
                    metric_name = full_name.split('{')[0]
                else:
                    metric_name = full_name
    
                if metric_name == 'vllm:prefix_cache_hits_total':
                    metrics['hits'] = value
                elif metric_name == 'vllm:prefix_cache_queries_total':
                    metrics['queries'] = value
                elif metric_name == 'vllm:kv_cache_usage_perc':
                    metrics['kv_cache_usage'] = value / 100.0
                    print(f"DEBUG: kv_cache_usage_perc raw={value}, converted={metrics['kv_cache_usage']}")
    
        except Exception as e:
            print(f"  [WARN] Failed to fetch metrics from port {port}: {e}")
    
        return metrics

    def _send_request(self, port: int, prompt: str, request_id: str) -> RequestMetrics:
        metrics = RequestMetrics(request_id=request_id, prompt=prompt, arrival_time=time.time())
        url = f"http://localhost:{port}/v1/completions"
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": self.max_tokens_per_request,
            "temperature": 0.0,
            "stream": True,
        }
        try:
            start = time.time()
            first_token_arrived = False
            token_count = 0
            with requests.post(url, json=payload, stream=True, timeout=self.timeout) as resp:
                if resp.status_code != 200:
                    metrics.status = "error"
                    return metrics
                for chunk in resp.iter_lines():
                    if chunk:
                        line = chunk.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                metrics.completion_time = time.time()
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices']:
                                    if not first_token_arrived:
                                        metrics.first_token_time = time.time()
                                        first_token_arrived = True
                                    token_count += 1
                            except:
                                pass
            if first_token_arrived and token_count > 0:
                metrics.total_tokens = token_count
                metrics.ttft = metrics.first_token_time - start
                metrics.tpot = (metrics.completion_time - metrics.first_token_time) / token_count
                metrics.status = "completed"
            else:
                metrics.status = "error"
        except Exception as e:
            print(f"  Request {request_id} error: {e}")
            metrics.status = "error"
        return metrics

    def _send_batch(self, port: int, prompts: List[str], concurrency: int) -> List[RequestMetrics]:
        results = []
        total = len(prompts)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for i, prompt in enumerate(prompts):
                req_id = f"req_{i:04d}"
                future = executor.submit(self._send_request, port, prompt, req_id)
                futures[future] = req_id
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 20 == 0:
                    print(f"    Progress: {completed}/{total}")
        return results

    def run_test(
        self,
        policy: str,
        prompts: List[str],
        concurrency: int,
        prefix_share_rate: float,
        test_name: str = "default"
    ) -> KVCacheTestResult:
        port = self.port_loss_aware if policy == "loss_aware" else self.port_lru
        print(f"  Sending {len(prompts)} requests to {policy.upper()} (port {port}) with concurrency={concurrency}...")
    
        # 1. 测试前抓取累计值
        before = self._fetch_metrics(port)
        hits_before = before['hits']
        queries_before = before['queries']


        
        start_time = time.time()
        results = self._send_batch(port, prompts, concurrency)
        end_time = time.time()
        total_time = end_time - start_time
    
        successful = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "error"]
        total_tokens = sum(r.total_tokens for r in successful)
        throughput = total_tokens / total_time if total_time > 0 else 0
    
        # 2. 测试后抓取累计值
        after = self._fetch_metrics(port)
        hits_after = after['hits']
        queries_after = after['queries']


        # 计算增量
        delta_hits = hits_after - hits_before
        delta_queries = queries_after - queries_before
        hit_rate = delta_hits / delta_queries if delta_queries > 0 else 0.0
    
        # 获取当前 KV 使用率（使用测试后的值）
        kv_usage = after['kv_cache_usage']
        
        short_ttfts = [r.ttft for r in successful if len(r.prompt) < 100 and r.ttft is not None]
        long_ttfts = [r.ttft for r in successful if len(r.prompt) >= 200 and r.ttft is not None]
    
        result = KVCacheTestResult(
            eviction_policy=policy,
            prefix_share_rate=prefix_share_rate,
            concurrency=concurrency,
            num_requests=len(results),
            successful=len(successful),
            failed=len(failed),
            ttfts=[r.ttft for r in successful if r.ttft is not None],
            tpots=[r.tpot for r in successful if r.tpot is not None],
            throughput=throughput,
            prefix_hit_rate=hit_rate,
            kv_cache_usage=kv_usage,
            num_evictions=0,  # 暂时无法从 metrics 获取
            short_ttfts=short_ttfts,
            long_ttfts=long_ttfts,
        )
    
        print(f"  ✓ Completed: {len(successful)} success, {len(failed)} failed")
        print(f"    Throughput: {throughput:.1f} tok/s")
        print(f"    Prefix Hit Rate: {hit_rate*100:.1f}%")
        print(f"    KV Usage: {kv_usage*100:.1f}%")
        print(f"    Evictions: {result.num_evictions}")
        if result.ttfts:
            print(f"    TTFT avg: {statistics.mean(result.ttfts)*1000:.2f} ms")
    
        return result


# ============================================================
# 4. 测试场景定义
# ============================================================

class TestScenarios:
    @staticmethod
    def scenarios():
        return [
            {'name': 'prefix_share_20', 'prefix_share_rate': 0.2, 'num_requests': 10},
            {'name': 'prefix_share_40', 'prefix_share_rate': 0.4, 'num_requests': 10},
            {'name': 'prefix_share_60', 'prefix_share_rate': 0.6, 'num_requests': 10},
            {'name': 'prefix_share_80', 'prefix_share_rate': 0.8, 'num_requests': 10},
        ]


# ============================================================
# 5. 报告生成器
# ============================================================

class KVCacheReporter:
    @staticmethod
    def generate_report(results: Dict[str, Dict[str, KVCacheTestResult]], output_file: str = "kv_cache_benchmark_report.json"):
        report = {'timestamp': datetime.now().isoformat(), 'tests': {}}
        for test_name, policy_results in results.items():
            report['tests'][test_name] = {}
            for policy, result in policy_results.items():
                if result:
                    report['tests'][test_name][policy] = result.compute_stats()
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
        return report

    @staticmethod
    def print_summary(results: Dict[str, Dict[str, KVCacheTestResult]]):
        print("\n" + "="*120)
        print("KV CACHE EVICTION BENCHMARK SUMMARY")
        print("="*120)
        headers = ["Test", "Policy", "Concurrency", "TTFT avg(ms)", "TTFT P95(ms)", "Hit Rate(%)", "Evictions", "Throughput(tok/s)"]
        print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]:>10} {headers[3]:>14} {headers[4]:>14} {headers[5]:>14} {headers[6]:>10} {headers[7]:>18}")
        print("-"*120)
        rows = []
        for test_name, policy_results in results.items():
            for policy, result in policy_results.items():
                if not result:
                    continue
                stats = result.compute_stats()
                rows.append({
                    'test': test_name,
                    'policy': policy,
                    'concurrency': result.concurrency,
                    'ttft_avg': stats.get('ttft_avg', 0),
                    'ttft_p95': stats.get('ttft_p95', 0),
                    'hit_rate': stats.get('prefix_hit_rate', 0) * 100,
                    'evictions': stats.get('num_evictions', 0),
                    'throughput': stats.get('throughput', 0),
                })
        for test_name in sorted(set(r['test'] for r in rows)):
            print(f"\n{test_name}:")
            for policy in ['lru', 'loss_aware']:
                sub = [r for r in rows if r['test'] == test_name and r['policy'] == policy]
                if not sub:
                    continue
                r = sub[0]
                print(f"  {policy.upper():<10} TTFT avg: {r['ttft_avg']:>6.1f}ms  P95: {r['ttft_p95']:>6.1f}ms  Hit: {r['hit_rate']:>5.1f}%  Evict: {r['evictions']:>6}  Throughput: {r['throughput']:>6.1f} tok/s")

    @staticmethod
    def plot_comparison(results: Dict[str, Dict[str, KVCacheTestResult]], output_file: str = "kv_cache_benchmark_plots.png"):
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed, skipping plots")
            return
        tests = list(results.keys())
        policies = ['lru', 'loss_aware']
        colors = {'lru': 'blue', 'loss_aware': 'green'}
        share_rates = []
        hit_rates_lru, hit_rates_loss = [], []
        evictions_lru, evictions_loss = [], []
        ttft_lru, ttft_loss = [], []
        for test in tests:
            rate = float(test.split('_')[-1]) / 100.0
            share_rates.append(rate)
            for policy in policies:
                result = results[test].get(policy)
                if result:
                    stats = result.compute_stats()
                    if policy == 'lru':
                        hit_rates_lru.append(stats.get('prefix_hit_rate', 0) * 100)
                        evictions_lru.append(stats.get('num_evictions', 0))
                        ttft_lru.append(stats.get('ttft_avg', 0))
                    else:
                        hit_rates_loss.append(stats.get('prefix_hit_rate', 0) * 100)
                        evictions_loss.append(stats.get('num_evictions', 0))
                        ttft_loss.append(stats.get('ttft_avg', 0))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(share_rates, hit_rates_lru, 'o-', color='blue', label='LRU')
        axes[0].plot(share_rates, hit_rates_loss, 's-', color='green', label='Loss-aware')
        axes[0].set_xlabel('Prefix Share Rate'); axes[0].set_ylabel('Hit Rate (%)'); axes[0].set_title('Prefix Cache Hit Rate'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(share_rates, evictions_lru, 'o-', color='blue', label='LRU')
        axes[1].plot(share_rates, evictions_loss, 's-', color='green', label='Loss-aware')
        axes[1].set_xlabel('Prefix Share Rate'); axes[1].set_ylabel('Evictions'); axes[1].set_title('Number of Evictions'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[2].plot(share_rates, ttft_lru, 'o-', color='blue', label='LRU')
        axes[2].plot(share_rates, ttft_loss, 's-', color='green', label='Loss-aware')
        axes[2].set_xlabel('Prefix Share Rate'); axes[2].set_ylabel('TTFT avg (ms)'); axes[2].set_title('Time to First Token'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(output_file, dpi=150); print(f"Plots saved to {output_file}"); plt.close()


# ============================================================
# 6. 主程序
# ============================================================

def main():
    MODEL_PATH = "/home/hx/Qwen2.5-1.5B-Instruct"
    PORT_LOSS_AWARE = 8000
    PORT_LRU = 8002
    CONCURRENCY = 16
    MAX_TOKENS_PER_REQUEST = 50

    print("="*100)
    print("VLLM KV CACHE EVICTION BENCHMARK (External Services)")
    print("="*100)
    print(f"Model: {MODEL_PATH}")
    print(f"Loss-aware port: {PORT_LOSS_AWARE}")
    print(f"LRU port: {PORT_LRU}")
    print(f"Concurrency: {CONCURRENCY}")
    print("="*100)

    engine = KVCacheBenchmarkEngine(
        model_path=MODEL_PATH,
        port_loss_aware=PORT_LOSS_AWARE,
        port_lru=PORT_LRU,
        max_tokens_per_request=MAX_TOKENS_PER_REQUEST
    )

    all_results = {}
    for scenario in TestScenarios.scenarios():
        test_name = scenario['name']
        prefix_share_rate = scenario['prefix_share_rate']
        num_requests = scenario['num_requests']
        print(f"\n{'#'*80}")
        print(f"Test: {test_name} (share={prefix_share_rate*100:.0f}%)")
        print("="*80)
        prompts = PrefixSharingDataset.generate_dataset(
            num_requests=num_requests,
            prefix_share_rate=prefix_share_rate,
            output_path=f"dataset_{test_name}.json"
        )
        test_results = {}
        for policy in ['lru', 'loss_aware']:
            print(f"\n--- Policy: {policy.upper()} ---")
            result = engine.run_test(
                policy=policy,
                prompts=prompts,
                concurrency=CONCURRENCY,
                prefix_share_rate=prefix_share_rate,
                test_name=test_name
            )
            test_results[policy] = result
        all_results[test_name] = test_results

    reporter = KVCacheReporter()
    reporter.generate_report(all_results)
    reporter.print_summary(all_results)
    reporter.plot_comparison(all_results)
    print("\n✅ All KV Cache benchmarks completed!")


if __name__ == "__main__":
    main()