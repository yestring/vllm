#!/usr/bin/env python3
"""
vLLM Scheduler 深度压力测试
测试维度：
1. 并发压力：1, 2, 4, 8, 16, 32, 64
2. 请求长度分布：短(10-50)、中(100-300)、长(500-1000)
3. 队头阻塞场景：混合长短请求（模拟真实混合负载）
4. 调度器极限：持续高压、突发流量、长尾延迟
"""
import subprocess
import threading
import time
import re
import time
import json
import requests
import statistics
import threading
import sys
import os
import subprocess
import random
import math
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
# 1. 提示词生成器 - 生成不同长度和复杂度的提示词
# ============================================================

class PromptGenerator:
    """生成不同长度和复杂度的测试提示词"""
    
    # 基础短句（用于生成不同长度）
    SHORT_SENTENCES = [
        "什么是人工智能？",
        "解释深度学习。",
        "Python和Java的区别。",
        "如何学习编程？",
        "今天天气怎么样？",
        "推荐一本好书。",
        "什么是机器学习？",
        "解释一下神经网络。",
        "如何开始学Python？",
        "什么是数据科学？",
    ]
    
    MEDIUM_TOPICS = [
        "Transformer模型的工作原理",
        "防止过拟合的方法",
        "大语言模型的训练流程",
        "CNN和ViT的对比",
        "梯度下降算法详解",
        "注意力机制的数学原理",
        "BERT和GPT的区别",
        "模型量化技术",
        "分布式训练策略",
        "多模态学习简介",
    ]
    
    LONG_TOPICS = [
        "从零开始实现一个Transformer模型，请详细解释每个组件的实现细节，包括多头注意力、前馈网络、层归一化、位置编码，并给出完整的PyTorch代码示例。",
        "请详细比较分析当前主流的大语言模型架构，包括GPT系列、LLaMA系列、Claude、Gemini等，从模型规模、训练数据、架构设计、性能表现等多个维度进行对比，并讨论各自的优缺点和适用场景。",
        "设计一个完整的推荐系统架构，包括数据收集、特征工程、召回策略（协同过滤、基于内容的推荐、向量召回）、排序模型（深度学习CTR模型）、在线服务架构、A/B测试框架，并给出关键模块的代码实现。",
        "请深入讲解联邦学习的技术原理、核心算法（FedAvg、FedProx、SCAFFOLD）、隐私保护机制（差分隐私、安全聚合）、通信优化策略，以及在实际业务场景中的应用案例和挑战。",
        "写一篇关于人工智能伦理的学术短文，涵盖算法偏见、数据隐私、就业影响、AI对齐、超级智能风险等核心议题，要求逻辑严密、论据充分，并给出你的个人观点和解决方案建议。",
    ]
    
    # 代码生成类（产生较长输出）
    CODE_PROMPTS = [
        "写一个完整的Python Web框架，包含路由、中间件、模板渲染、ORM集成，要求代码结构清晰，有完整的注释和文档。",
        "实现一个支持并发请求的HTTP服务器，使用Python的asyncio，包含连接池、超时处理、请求解析、响应构建，要求代码可运行。",
        "设计一个分布式任务调度系统，包含任务队列、Worker管理、失败重试、任务依赖、监控告警，给出核心代码和架构图。",
    ]
    
    @classmethod
    def generate_short(cls, count: int) -> List[str]:
        """生成短提示词 (10-50 tokens)"""
        prompts = []
        for _ in range(count):
            base = random.choice(cls.SHORT_SENTENCES)
            # 随机加一些变化
            if random.random() > 0.5:
                base += "？请简要回答。"
            prompts.append(base)
        return prompts
    
    @classmethod
    def generate_medium(cls, count: int) -> List[str]:
        """生成中等长度提示词 (100-300 tokens)"""
        prompts = []
        for _ in range(count):
            topic = random.choice(cls.MEDIUM_TOPICS)
            # 随机添加要求
            requirements = [
                f"请详细解释{topic}，包括核心概念和实际应用。",
                f"请从理论和实践两个角度分析{topic}。",
                f"请用通俗易懂的语言解释{topic}，并给出一个例子。",
                f"请全面阐述{topic}，包括发展历程和未来趋势。",
            ]
            prompts.append(random.choice(requirements))
        return prompts
    
    @classmethod
    def generate_long(cls, count: int) -> List[str]:
        """生成长提示词 (500-1000 tokens)"""
        prompts = []
        for _ in range(count):
            if random.random() > 0.6:
                prompts.append(random.choice(cls.LONG_TOPICS))
            else:
                prompts.append(random.choice(cls.CODE_PROMPTS))
        return prompts
    
    @classmethod
    def generate_mixed(cls, total: int, 
                       short_ratio: float = 0.4,
                       medium_ratio: float = 0.4,
                       long_ratio: float = 0.2) -> List[str]:
        """生成混合长度的提示词，模拟真实负载"""
        short_count = int(total * short_ratio)
        medium_count = int(total * medium_ratio)
        long_count = total - short_count - medium_count
        
        prompts = []
        prompts.extend(cls.generate_short(short_count))
        prompts.extend(cls.generate_medium(medium_count))
        prompts.extend(cls.generate_long(long_count))
        
        # 打乱顺序，模拟随机到达
        random.shuffle(prompts)
        return prompts


# ============================================================
# 2. 测试结果数据结构
# ============================================================

@dataclass
class RequestMetrics:
    """单个请求的指标"""
    request_id: str
    prompt_length: int
    arrival_time: float
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    total_tokens: int = 0
    ttft: Optional[float] = None
    tpot: Optional[float] = None
    completion_latency: Optional[float] = None
    status: str = "pending"


@dataclass
class StressTestResult:
    """压力测试结果"""
    policy: str
    concurrency: int
    test_name: str
    
    # 请求统计
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # 性能指标
    throughput: float = 0.0  # tok/s
    ttfts: List[float] = field(default_factory=list)
    tpots: List[float] = field(default_factory=list)
    completion_latencies: List[float] = field(default_factory=list)
    
    # 按长度分组
    short_ttfts: List[float] = field(default_factory=list)
    medium_ttfts: List[float] = field(default_factory=list)
    long_ttfts: List[float] = field(default_factory=list)
    
    # 队头阻塞指标
    head_of_line_blocking_ratio: float = 0.0
    starvation_count: int = 0
    
    # 时间线（用于可视化）
    timeline: List[Tuple[float, float]] = field(default_factory=list)  # (time, ttft)
    
    def compute_stats(self) -> Dict[str, Any]:
        """计算统计指标"""
        stats = {}
        
        # 基础统计
        stats['total_requests'] = self.total_requests
        stats['successful'] = self.successful_requests
        stats['failed'] = self.failed_requests
        stats['success_rate'] = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
        
        # 吞吐量
        stats['throughput'] = self.throughput
        
        # TTFT
        if self.ttfts:
            stats['ttft_avg'] = statistics.mean(self.ttfts) * 1000
            stats['ttft_p50'] = statistics.median(self.ttfts) * 1000
            if len(self.ttfts) >= 20:
                stats['ttft_p95'] = np.percentile(self.ttfts, 95) * 1000
                stats['ttft_p99'] = np.percentile(self.ttfts, 99) * 1000
            else:
                stats['ttft_p95'] = max(self.ttfts) * 1000
                stats['ttft_p99'] = max(self.ttfts) * 1000
            stats['ttft_std'] = statistics.stdev(self.ttfts) * 1000 if len(self.ttfts) > 1 else 0
        
        # 按长度分组的 TTFT
        for name, data in [('short', self.short_ttfts), 
                           ('medium', self.medium_ttfts), 
                           ('long', self.long_ttfts)]:
            if data:
                stats[f'ttft_{name}_avg'] = statistics.mean(data) * 1000
                stats[f'ttft_{name}_p95'] = np.percentile(data, 95) * 1000 if len(data) >= 20 else max(data) * 1000
        
        # 长度差异比（长请求 vs 短请求）
        if self.short_ttfts and self.long_ttfts:
            stats['short_vs_long_ratio'] = statistics.mean(self.long_ttfts) / statistics.mean(self.short_ttfts)
        
        # TPOT
        if self.tpots:
            stats['tpot_avg'] = statistics.mean(self.tpots) * 1000
        
        # Completion Latency
        if self.completion_latencies:
            stats['latency_avg'] = statistics.mean(self.completion_latencies)
            stats['latency_p99'] = np.percentile(self.completion_latencies, 99) if len(self.completion_latencies) >= 20 else max(self.completion_latencies)
        
        # 队头阻塞指标
        stats['head_of_line_blocking_ratio'] = self.head_of_line_blocking_ratio
        stats['starvation_count'] = self.starvation_count
        
        return stats


# ============================================================
# 3. 压力测试引擎
# ============================================================

class StressTestEngine:
    """压力测试引擎"""
    
    def __init__(self, model_path: str,
                 ports: Dict[str, int],
                 max_model_len: int = 2048,
                 gpu_device: str = "0",
                 max_tokens_per_request: int = 50):
        
        self.model_path = model_path
        self.ports = ports
        self.max_model_len = max_model_len
        self.gpu_device = gpu_device
        self.max_tokens_per_request = max_tokens_per_request
        self.process = None
        
    def _start_server(self, policy: str) -> bool:
        """启动 vLLM 服务"""
        port = self.ports[policy]
        
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--scheduling-policy", policy,
            "--port", str(port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", "0.85",
            "--max-num-seqs", "128",  # 更大的批次
            "--max-num-batched-tokens", "4096",
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_device
        
        print(f"  Starting {policy.upper()} server on port {port}...")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        
        # 等待启动
        timeout = 180
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=2)
                if resp.status_code == 200:
                    print(f"  ✓ Server started (took {time.time()-start_time:.1f}s)")
                    return True
            except:
                pass
            time.sleep(1)
        
        print(f"  ✗ Server failed to start")
        return False
    
    def _stop_server(self):
        """停止服务"""
        if self.process:
            self.process.terminate()
            time.sleep(3)
            if self.process.poll() is None:
                self.process.kill()
            self.process = None
            time.sleep(3)
    
    def _send_request(self, port: int, prompt: str, 
                      request_id: str) -> RequestMetrics:
        """发送单个请求"""
        metrics = RequestMetrics(
            request_id=request_id,
            prompt_length=len(prompt),
            arrival_time=time.time()
        )
        
        url = f"http://localhost:{port}/v1/completions"
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": self.max_tokens_per_request,
            "stream": True,
            "temperature": 0.0
        }
        
        try:
            start = time.time()
            first_token_arrived = False
            token_count = 0
            
            with requests.post(url, json=payload, stream=True, timeout=180) as resp:
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
                metrics.completion_latency = metrics.completion_time - start
                metrics.status = "completed"
            else:
                metrics.status = "error"
                
        except Exception as e:
            print(f"  Request {request_id} error: {e}")
            metrics.status = "error"
        
        return metrics
    
    def _send_burst_requests(self, port: int, prompts: List[str],
                             concurrency: int) -> List[RequestMetrics]:
        """突发发送一批请求"""
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
    
    def run_test(self, policy: str, concurrency: int,
                 prompts: List[str], test_name: str = "default") -> StressTestResult:
        """运行单个压力测试"""
        port = self.ports[policy]
        gpu_monitor = GPUMonitor(gpu_id=1, interval=0.5)
        gpu_monitor.start()
        if not self._start_server(policy):
            return StressTestResult(
                policy=policy, concurrency=concurrency,
                test_name=test_name,
                total_requests=0, successful_requests=0, failed_requests=0
            )
        
        try:
            print(f"  Running {len(prompts)} requests with concurrency={concurrency}...")
            
            start_time = time.time()
            results = self._send_burst_requests(port, prompts, concurrency)
            end_time = time.time()
            total_time = end_time - start_time
            
            successful = [r for r in results if r.status == "completed"]
            failed = [r for r in results if r.status == "error"]
            
            total_tokens = sum(r.total_tokens for r in successful)
            throughput = total_tokens / total_time if total_time > 0 else 0
            
            # 按长度分组
            short_ttfts = [r.ttft for r in successful if r.prompt_length < 50 and r.ttft is not None]
            medium_ttfts = [r.ttft for r in successful if 50 <= r.prompt_length < 200 and r.ttft is not None]
            long_ttfts = [r.ttft for r in successful if r.prompt_length >= 200 and r.ttft is not None]
            
            result = StressTestResult(
                policy=policy,
                concurrency=concurrency,
                test_name=test_name,
                total_requests=len(results),
                successful_requests=len(successful),
                failed_requests=len(failed),
                throughput=throughput,
                ttfts=[r.ttft for r in successful if r.ttft is not None],
                tpots=[r.tpot for r in successful if r.tpot is not None],
                completion_latencies=[r.completion_latency for r in successful if r.completion_latency is not None],
                short_ttfts=short_ttfts,
                medium_ttfts=medium_ttfts,
                long_ttfts=long_ttfts,
            )
            
            # 计算队头阻塞指标：长请求与短请求的 TTFT 比值
            if short_ttfts and long_ttfts:
                result.head_of_line_blocking_ratio = statistics.mean(long_ttfts) / statistics.mean(short_ttfts)
            
            print(f"  ✓ Completed: {len(successful)} success, {len(failed)} failed")
            print(f"    Throughput: {throughput:.1f} tok/s")
            print(f"    TTFT avg: {statistics.mean(result.ttfts)*1000:.2f} ms" if result.ttfts else "    TTFT: N/A")


             # 在服务停止前，从日志中解析调度器统计
            scheduler_stats = self._parse_scheduler_stats()  # 见下方
            
            gpu_monitor.stop()
            gpu_stats = gpu_monitor.get_stats()
            
            # 将 scheduler_stats 和 gpu_stats 存入 result
            result.scheduler_overheads = [scheduler_stats.get('avg_total_ms', 0)]
            result.gpu_utilizations = [gpu_stats.get('gpu_util_avg', 0)]


            
            return result
            
        finally:
            self._stop_server()


    def _parse_scheduler_stats(self):
        # 如果在启动时把 stderr 重定向到文件，可以读取文件
        # 简单起见，可以修改 vLLM 让调度器把统计写到临时 JSON 文件
        # 这里给出一个占位实现
        return {'avg_total_ms': 0.5, 'avg_score_ms': 0.2, 'avg_sort_ms': 0.1}

# ============================================================
# 4. 测试场景定义
# ============================================================

class TestScenarios:
    """定义各种压力测试场景"""
    
    @staticmethod
    def scenarios():
        return [
            {
                'name': 'high_concurrency',
                'description': '高并发纯短请求',
                'prompt_generator': lambda: PromptGenerator.generate_short(2000),
                'concurrency_levels': [],
            },
            {
                'name': 'mixed_length',
                'description': '混合长度（20%短 + 40%中 + 40%长）',
                'prompt_generator': lambda: PromptGenerator.generate_mixed(2000, 0.2, 0.4, 0.4),
                'concurrency_levels': [],
            },
            {
                'name': 'head_of_line_blocking',
                'description': '队头阻塞测试（10%超长 + 90%超短）',
                'prompt_generator': lambda: PromptGenerator.generate_mixed(2000, 0.8, 0.1, 0.1),
                'concurrency_levels': [8,16,32,64,128,256],
            },
            {
                'name': 'extreme_long',
                'description': '极端长请求（全部长/代码请求）',
                'prompt_generator': lambda: PromptGenerator.generate_long(2000),
                'concurrency_levels': [],
            },
        ]


# ============================================================
# 5. 报告生成器
# ============================================================

class StressTestReporter:
    """压力测试报告生成器"""
    
    @staticmethod
    def generate_report(results: Dict[str, Dict[str, Dict[int, StressTestResult]]],
                       output_file: str = "stress_test_report.json"):
        """
        生成 JSON 报告
        
        Args:
            results: {test_name: {policy: {concurrency: StressTestResult}}}
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        for test_name, policy_results in results.items():
            report['tests'][test_name] = {}
            for policy, conv_results in policy_results.items():
                report['tests'][test_name][policy] = {}
                for concurrency, result in conv_results.items():
                    if result and hasattr(result, 'compute_stats'):
                        stats = result.compute_stats()
                        report['tests'][test_name][policy][str(concurrency)] = stats
                    else:
                        report['tests'][test_name][policy][str(concurrency)] = {
                            'error': 'Invalid result object'
                        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
        return report
    
    @staticmethod
    def print_summary(results: Dict[str, Dict[str, Dict[int, StressTestResult]]]):
        """打印汇总结果"""
        print("\n" + "="*120)
        print("STRESS TEST SUMMARY")
        print("="*120)
        
        # 定义要显示的指标
        metrics_to_show = [
            ('throughput', 'Throughput', '{:.1f} tok/s'),
            ('ttft_avg', 'TTFT avg', '{:.2f} ms'),
            ('ttft_p95', 'TTFT P95', '{:.2f} ms'),
            ('ttft_p99', 'TTFT P99', '{:.2f} ms'),
            ('success_rate', 'Success Rate', '{:.1%}'),
        ]
        
        for test_name, policy_results in results.items():
            print(f"\n📊 Test: {test_name}")
            print("-"*120)
            
            policies = list(policy_results.keys())
            
            # 打印表头
            print(f"{'Concurrency':>12} |", end="")
            for policy in policies:
                print(f" {policy.upper():>18} |", end="")
            print()
            print("-"*120)
            
            # 获取所有并发级别
            all_convs = set()
            for policy in policies:
                all_convs.update(policy_results[policy].keys())
            conv_levels = sorted(all_convs)
            
            # 对每个并发级别打印数据
            for conv in conv_levels:
                print(f"{conv:>12} |", end="")
                for policy in policies:
                    result = policy_results[policy].get(conv)
                    if result and hasattr(result, 'compute_stats'):
                        stats = result.compute_stats()
                        ttft = stats.get('ttft_avg', 0)
                        throughput = stats.get('throughput', 0)
                        success_rate = stats.get('success_rate', 0)
                        print(f" {throughput:>5.1f}/{ttft:>5.1f}ms/{success_rate:>5.1%} |", end="")
                    else:
                        print(f" {'N/A':>18} |", end="")
                print()
        
        # 额外打印：队头阻塞指标
        print("\n" + "-"*120)
        print("HEAD-OF-LINE BLOCKING METRICS (Long/Short TTFT Ratio, lower is better)")
        print("-"*80)
        
        for test_name, policy_results in results.items():
            print(f"\n{test_name}:")
            for policy, conv_results in policy_results.items():
                ratios = []
                for conv, result in conv_results.items():
                    if result and hasattr(result, 'compute_stats'):
                        if result.head_of_line_blocking_ratio > 0:
                            ratios.append(result.head_of_line_blocking_ratio)
                if ratios:
                    print(f"  {policy.upper()}: avg={statistics.mean(ratios):.2f}, "
                          f"min={min(ratios):.2f}, max={max(ratios):.2f}")
    
    @staticmethod
    def plot_comparison(results: Dict[str, Dict[str, Dict[int, StressTestResult]]],
                        output_file: str = "stress_test_plots.png"):
        """生成对比图表"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed, skipping plots")
            return
        
        tests = list(results.keys())
        policies = ['fcfs', 'utility']
        colors = {'fcfs': 'blue', 'utility': 'green'}
        markers = {'fcfs': 'o', 'utility': 's'}
        
        # 计算需要的子图数量
        n_tests = min(len(tests), 3)
        
        fig, axes = plt.subplots(2, n_tests, figsize=(6*n_tests, 10))
        if n_tests == 1:
            axes = axes.reshape(2, 1)
        
        for idx, test_name in enumerate(tests[:n_tests]):
            # TTFT vs Concurrency
            ax = axes[0, idx]
            for policy in policies:
                if policy in results[test_name]:
                    convs = sorted(results[test_name][policy].keys())
                    vals = []
                    for c in convs:
                        r = results[test_name][policy][c]
                        if r and hasattr(r, 'compute_stats'):
                            stats = r.compute_stats()
                            vals.append(stats.get('ttft_avg', 0))
                    if vals:
                        ax.plot(convs[:len(vals)], vals, 
                               marker=markers[policy], color=colors[policy], 
                               label=policy.upper(), linewidth=2, markersize=8)
            ax.set_xlabel('Concurrency')
            ax.set_ylabel('TTFT avg (ms)')
            ax.set_title(f'{test_name} - TTFT')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Throughput vs Concurrency
            ax = axes[1, idx]
            for policy in policies:
                if policy in results[test_name]:
                    convs = sorted(results[test_name][policy].keys())
                    vals = []
                    for c in convs:
                        r = results[test_name][policy][c]
                        if r and hasattr(r, 'compute_stats'):
                            stats = r.compute_stats()
                            vals.append(stats.get('throughput', 0))
                    if vals:
                        ax.plot(convs[:len(vals)], vals,
                               marker=markers[policy], color=colors[policy],
                               label=policy.upper(), linewidth=2, markersize=8)
            ax.set_xlabel('Concurrency')
            ax.set_ylabel('Throughput (tok/s)')
            ax.set_title(f'{test_name} - Throughput')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 如果有第4个测试，单独画一列
        if len(tests) > n_tests:
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
            for test_name in tests[n_tests:]:
                for policy in policies:
                    if policy in results[test_name]:
                        convs = sorted(results[test_name][policy].keys())
                        vals = []
                        for c in convs:
                            r = results[test_name][policy][c]
                            if r and hasattr(r, 'compute_stats'):
                                stats = r.compute_stats()
                                vals.append(stats.get('ttft_avg', 0))
                        if vals:
                            label = f"{test_name} ({policy.upper()})"
                            ax2.plot(convs[:len(vals)], vals,
                                    marker=markers[policy], color=colors[policy],
                                    label=label, linewidth=2, markersize=8)
            ax2.set_xlabel('Concurrency')
            ax2.set_ylabel('TTFT avg (ms)')
            ax2.set_title('Additional Tests - TTFT')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(output_file.replace('.png', '_extra.png'), dpi=150)
            print(f"Extra plots saved to {output_file.replace('.png', '_extra.png')}")
            plt.close(fig2)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Plots saved to {output_file}")
        plt.close()




class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.5):
        self.gpu_id = gpu_id
        self.interval = interval
        self.running = False
        self.samples = []
        self.thread = None

    def start(self):
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _monitor(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "--id", str(self.gpu_id),
            "-l", str(self.interval)
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while self.running:
            line = proc.stdout.readline()
            if not line:
                break
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 4:
                self.samples.append({
                    'timestamp': time.time(),
                    'gpu_util': float(parts[0]),
                    'mem_util': float(parts[1]),
                    'mem_used_mb': float(parts[2]),
                    'mem_total_mb': float(parts[3])
                })
        proc.terminate()

    def get_stats(self):
        if not self.samples:
            return {}
        gpu_utils = [s['gpu_util'] for s in self.samples]
        mem_utils = [s['mem_util'] for s in self.samples]
        return {
            'gpu_util_avg': sum(gpu_utils) / len(gpu_utils),
            'gpu_util_min': min(gpu_utils),
            'gpu_util_max': max(gpu_utils),
            'mem_util_avg': sum(mem_utils) / len(mem_utils),
        }









    
# ============================================================
# 6. 主程序
# ============================================================

def main():
    # 配置
    MODEL_PATH = "/home/hx/Qwen2.5-1.5B-Instruct"
    PORTS = {
        'fcfs': 8002,
        'utility': 8000
    }
    GPU_DEVICE = "1"
    MAX_MODEL_LEN = 2048
    MAX_TOKENS_PER_REQUEST = 60
    
    print("="*100)
    print("VLLM SCHEDULER STRESS TEST")
    print("="*100)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {GPU_DEVICE}")
    print(f"Ports: {PORTS}")
    print(f"Max model len: {MAX_MODEL_LEN}")
    print(f"Max tokens per request: {MAX_TOKENS_PER_REQUEST}")
    print("="*100)
    
    # 初始化测试引擎
    engine = StressTestEngine(
        model_path=MODEL_PATH,
        ports=PORTS,
        max_model_len=MAX_MODEL_LEN,
        gpu_device=GPU_DEVICE,
        max_tokens_per_request=MAX_TOKENS_PER_REQUEST
    )
    
     # 运行所有测试场景
    all_results = {}  # test_name -> {policy: {concurrency: StressTestResult}}
    
    for scenario in TestScenarios.scenarios():
        test_name = scenario['name']
        description = scenario['description']
        prompt_generator = scenario['prompt_generator']
        concurrency_levels = scenario['concurrency_levels']
        
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"Description: {description}")
        print(f"Concurrency: {concurrency_levels}")
        print("="*80)
        
        # 生成提示词
        prompts = prompt_generator()
        print(f"Generated {len(prompts)} prompts")
        
        test_results = {}
        
        for policy in ['fcfs', 'utility']:
            print(f"\n--- Testing {policy.upper()} ---")
            
            policy_results = {}
            for concurrency in concurrency_levels:
                print(f"\nConcurrency: {concurrency}")
                result = engine.run_test(policy, concurrency, prompts, test_name)
                policy_results[concurrency] = result
            
            test_results[policy] = policy_results
        
        all_results[test_name] = test_results
    
    # 生成报告（传入正确的数据结构）
    reporter = StressTestReporter()
    reporter.generate_report(all_results)  # all_results 是 {test_name: {policy: {concurrency: StressTestResult}}}
    reporter.print_summary(all_results)
    reporter.plot_comparison(all_results)
    
    print("\n✅ All stress tests completed!")


if __name__ == "__main__":
    main()