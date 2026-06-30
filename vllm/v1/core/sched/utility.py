# vllm/v1/core/sched/utility_scheduler.py

class UtilityScheduler:
    def __init__(self, config: UtilityConfig):
        self.config = config
        
    def compute_utility(self, request: Request, now: float) -> float:
        """效用计算，由 request 的 utility_score 实现"""
        return request.utility_score(self.config.weights, now)
    
    def estimate_cost(self, request: Request) -> float:
        """估算资源消耗"""
        if request.num_computed_tokens < request.num_prompt_tokens:
            return request.num_prompt_tokens - request.num_computed_tokens
        return 1.0
    
    def select_batch(self, requests: list[Request], 
                    token_budget: int, 
                    kv_budget: int) -> list[Request]:
        """贪婪选择：按 utility/cost 排序"""
        now = time.monotonic()
        scored = []
        for req in requests:
            utility = self.compute_utility(req, now)
            cost = self.estimate_cost(req)
            if cost <= 0:
                continue
            scored.append((req, utility / cost))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        used_tokens = 0
        used_kv = 0
        for req, density in scored:
            cost = self.estimate_cost(req)
            if used_tokens + cost <= token_budget:
                selected.append(req)
                used_tokens += cost
        return selected