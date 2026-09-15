"""
PRODUCTION CONTEXT SYSTEM
Complete production-ready context pipeline

Run: python production_system.py
"""

import time
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

# Import from our modules (in production, these would be proper imports)
from context_cache import MultiLayerCache
from rate_limiter import MultiTierRateLimiter
from cost_manager import CostTracker, BudgetEnforcer
from graceful_degradation import GracefulDegrader, DegradationLevel
from retry_patterns import RobustRetry, CircuitOpenError

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ProductionConfig:
    # Rate limiting
    global_rpm: int = 100
    user_rpm: int = 20
    tokens_per_minute: int = 100000

    # Caching
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 1000

    # Context assembly (token budget per degradation tier)
    full_context_tokens: int = 400
    reduced_context_tokens: int = 40

    # Cost management
    daily_budget: float = 100.0
    user_daily_budget: float = 5.0

    # Retry settings
    max_retries: int = 3
    base_delay: float = 0.5

    # Circuit breaker
    failure_threshold: int = 5


# =============================================================================
# MOCK LLM CLIENT
# =============================================================================


class MockLLMClient:
    """Simulates LLM API with realistic behavior."""

    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate
        self.call_count = 0

    def generate(self, query: str, context: str) -> Dict:
        """Simulate LLM generation."""
        self.call_count += 1

        # Simulate occasional failures
        if random.random() < self.failure_rate:
            raise Exception("Rate limit exceeded - 429")

        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))

        input_tokens = len(query.split()) + len(context.split()) * 2
        output_tokens = random.randint(100, 500)

        return {
            "text": f"Response to: {query[:50]}...",
            "model": "claude-sonnet-5",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


# =============================================================================
# CONTEXT ASSEMBLER
# =============================================================================


class ContextAssembler:
    """Builds the context we actually send to the model.

    Rank candidate snippets against the query, then pack as many as the token
    budget allows. Shrinking that budget is how we degrade under load:
    FULL (rich context) -> REDUCED (one tight snippet) -> MINIMAL (no context).
    This is the context-engineering core the whole production system wraps.
    """

    KNOWLEDGE = [
        "Machine learning trains models on data to make predictions without explicit rules.",
        "Neural networks stack layers of weighted connections that learn features from examples.",
        "Deep learning uses many-layered neural networks to model complex patterns.",
        "Transformers use self-attention to weigh how tokens relate across a sequence.",
        "Attention lets a model focus on the most relevant parts of its input.",
        "Backpropagation adjusts weights by pushing error gradients back through the layers.",
        "Gradient descent iteratively minimizes a loss function to fit model parameters.",
        "Embeddings map words and tokens into vectors that capture their meaning.",
        "A context window is the fixed span of tokens a model can attend to at once.",
        "Overfitting is when a model memorizes training data instead of generalizing.",
    ]

    @staticmethod
    def count_tokens(text: str) -> int:
        """Rough token estimate (words x 1.3)."""
        return int(len(text.split()) * 1.3) if text else 0

    def assemble(self, query: str, token_budget: int, max_snippets: int = 4) -> str:
        """Rank snippets by word overlap with the query, then pack to budget."""
        q_words = set(query.lower().replace("?", "").split())
        ranked = sorted(
            self.KNOWLEDGE,
            key=lambda s: len(q_words & set(s.lower().split())),
            reverse=True,
        )

        chosen, used = [], 0
        for snippet in ranked[:max_snippets]:
            cost = self.count_tokens(snippet)
            if used + cost > token_budget:
                break
            chosen.append(snippet)
            used += cost

        return "\n".join(chosen)


# =============================================================================
# PRODUCTION CONTEXT SYSTEM
# =============================================================================


class ProductionContextSystem:
    """Complete production context system."""

    def __init__(self, config: ProductionConfig = None):
        self.config = config or ProductionConfig()

        # Initialize components
        self.assembler = ContextAssembler()
        self.cache = MultiLayerCache(l1_max_size=100)
        self.rate_limiter = MultiTierRateLimiter(
            global_rpm=self.config.global_rpm,
            user_rpm=self.config.user_rpm,
            tokens_per_minute=self.config.tokens_per_minute,
        )
        self.cost_tracker = CostTracker()
        self.budget_enforcer = BudgetEnforcer(self.cost_tracker)
        self.retry = RobustRetry(
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            failure_threshold=self.config.failure_threshold,
        )
        self.degrader = GracefulDegrader()

        # Mock client (higher failure rate so retries + degradation are visible)
        self.llm_client = MockLLMClient(failure_rate=0.4)

        # Setup budgets
        self.cost_tracker.set_budget("global", self.config.daily_budget)

        # Setup degradation
        self._setup_degradation()

        # Stats
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "rate_limited": 0,
            "budget_exceeded": 0,
            "errors": 0,
            "degraded": 0,
            "retries": 0,
        }

    def _setup_degradation(self):
        """Setup degradation levels."""
        self.degrader.add_step(
            DegradationLevel.FULL,
            "full_pipeline",
            self._full_pipeline,
            "Complete context pipeline",
        )
        self.degrader.add_step(
            DegradationLevel.REDUCED,
            "reduced_context",
            self._reduced_pipeline,
            "Simplified context",
        )
        self.degrader.add_step(
            DegradationLevel.MINIMAL,
            "minimal",
            self._minimal_pipeline,
            "Direct response",
        )

    def process(self, user_id: str, query: str) -> Dict:
        """Process a request through the full pipeline."""
        start_time = time.time()
        self.stats["requests"] += 1

        # Step 1: Rate limit check
        rate_check = self.rate_limiter.check(user_id, estimated_tokens=5000)
        if not rate_check["allowed"]:
            self.stats["rate_limited"] += 1
            return self._rate_limited_response(rate_check)

        # Step 2: Assemble the context — the context-engineering core. The
        # assembled context (not the raw query) is the request's real identity.
        context = self.assembler.assemble(
            query, token_budget=self.config.full_context_tokens
        )

        # Step 3: Cache check — keyed on the ASSEMBLED CONTEXT via a fingerprint,
        # so when the underlying knowledge changes, the key changes and stale
        # answers invalidate themselves.
        cache_key = self._context_key(user_id, query, context)
        cached = self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            return {
                "success": True,
                "response": cached,
                "cache_hit": True,
                "latency_ms": (time.time() - start_time) * 1000,
            }

        # Step 4: Budget check
        user_key = f"user:{user_id}"
        if user_key not in self.cost_tracker.budgets:
            self.cost_tracker.set_budget(user_key, self.config.user_daily_budget)

        budget_check = self.budget_enforcer.check_budget(user_key)
        if not budget_check["allowed"]:
            self.stats["budget_exceeded"] += 1
            return self._budget_exceeded_response(budget_check)

        # Step 5: Execute pipeline with retry + degradation.
        # The degrader is the OUTER fallback (FULL -> REDUCED -> MINIMAL); retry
        # lives INSIDE each tier around the LLM call. Transient errors are retried
        # first; only when retries are exhausted does the tier fail and degrade.
        # Each tier assembles its OWN context at its budget, so degrading = sending
        # less context (FULL -> REDUCED -> MINIMAL), not just swapping models.
        retries_before = self.stats["retries"]
        try:
            result, level = self.degrader.execute(query)

            if level != DegradationLevel.FULL:
                self.stats["degraded"] += 1

        except CircuitOpenError:
            self.stats["errors"] += 1
            return self._circuit_open_response()
        except Exception as e:
            self.stats["errors"] += 1
            return self._error_response(e)

        # Step 6: Cache the response (under its assembled-context key)
        if result:
            self.cache.set(cache_key, result["response"])

        # Step 7: Record costs
        if result:
            self.cost_tracker.record(
                operation="query",
                model=result.get("model", "claude-sonnet-5"),
                input_tokens=result.get("input_tokens", 1000),
                output_tokens=result.get("output_tokens", 200),
                user_id=user_id,
            )
            self.rate_limiter.record_tokens(
                result.get("input_tokens", 1000) + result.get("output_tokens", 200)
            )

        # Step 8: Return with metadata
        latency = (time.time() - start_time) * 1000

        return {
            "success": True,
            "response": result["response"] if result else "Processing failed",
            "cache_hit": False,
            "latency_ms": latency,
            "degradation_level": level.name if level else "UNKNOWN",
            "retries": self.stats["retries"] - retries_before,
            "context_tokens": result.get("context_tokens", 0) if result else 0,
        }

    def _call_llm_with_retry(self, query: str, context: str) -> Dict:
        """Call the LLM through the retry + circuit breaker, counting retries."""
        attempts = {"n": 0}

        def _attempt():
            attempts["n"] += 1
            return self.llm_client.generate(query, context)

        try:
            return self.retry.execute(_attempt)
        finally:
            # Every attempt past the first is a retry
            self.stats["retries"] += max(0, attempts["n"] - 1)

    def _context_key(self, user_id: str, query: str, context: str) -> str:
        """Cache identity = the ASSEMBLED CONTEXT (fingerprinted) plus the query it
        answers. Keying on the context means a changed knowledge base changes the
        key, so stale answers invalidate themselves."""
        fingerprint = hashlib.sha1(context.encode()).hexdigest()[:8]
        return f"{user_id}:{query}:ctx-{fingerprint}"

    def _package(self, response: Dict, context: str) -> Dict:
        """Wrap an LLM response with the context size that produced it."""
        return {
            "response": response["text"],
            "model": response["model"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "context_tokens": self.assembler.count_tokens(context),
        }

    def _full_pipeline(self, query: str) -> Dict:
        """FULL: rich assembled context."""
        context = self.assembler.assemble(
            query, token_budget=self.config.full_context_tokens
        )
        response = self._call_llm_with_retry(query, context)
        return self._package(response, context)

    def _reduced_pipeline(self, query: str) -> Dict:
        """REDUCED: trim the context to a tight budget (one best snippet)."""
        context = self.assembler.assemble(
            query, token_budget=self.config.reduced_context_tokens, max_snippets=1
        )
        response = self._call_llm_with_retry(query, context)
        return self._package(response, context)

    def _minimal_pipeline(self, query: str) -> Dict:
        """MINIMAL: no context at all — cheapest possible answer, always succeeds."""
        return {
            "response": "I'm currently experiencing high demand. Please try again.",
            "model": "fallback",
            "input_tokens": 0,
            "output_tokens": 0,
            "context_tokens": 0,
        }

    def _rate_limited_response(self, check: Dict) -> Dict:
        return {
            "success": False,
            "error": "rate_limited",
            "message": "Too many requests",
            "retry_after": check.get("retry_after", 60),
        }

    def _budget_exceeded_response(self, check: Dict) -> Dict:
        return {
            "success": False,
            "error": "budget_exceeded",
            "message": "Daily limit reached",
            "usage": check.get("usage", 1.0),
        }

    def _circuit_open_response(self) -> Dict:
        return {
            "success": False,
            "error": "service_unavailable",
            "message": "Service temporarily unavailable",
        }

    def _error_response(self, error: Exception) -> Dict:
        return {
            "success": False,
            "error": "internal_error",
            "message": str(error)[:100],
        }

    def get_dashboard(self) -> Dict:
        """Get system status dashboard."""
        return {
            "health": self._get_health(),
            "stats": self.stats,
            "cache": self.cache.get_stats(),
            "rate_limits": self.rate_limiter.get_status(),
            "costs": self.cost_tracker.get_summary("day"),
            "circuit": self.retry.get_status(),
        }

    def _get_health(self) -> str:
        """Determine overall health."""
        circuit_state = self.retry.circuit.state.value
        if circuit_state == "open":
            return "DEGRADED"
        if circuit_state == "half_open":
            return "RECOVERING"
        if self.stats["errors"] / max(1, self.stats["requests"]) > 0.1:
            return "DEGRADED"
        return "HEALTHY"


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Seed the RNG so the demo is reproducible: the failures, latencies, and
    # traffic mix below are the SAME every run. No API key needed — the LLM is
    # mocked (MockLLMClient), so this whole pipeline runs fully offline.
    random.seed(22)

    print("=" * 60)
    print("PRODUCTION CONTEXT SYSTEM")
    print("=" * 60)

    config = ProductionConfig(
        global_rpm=60,
        user_rpm=15,
        daily_budget=10.0,
        user_daily_budget=2.0,
        max_retries=2,
        base_delay=0.3,
    )

    system = ProductionContextSystem(config)

    # Simulate traffic
    print("\n" + "-" * 50)
    print("SIMULATING TRAFFIC (45 requests)")
    print("-" * 50)

    users = ["user_a", "user_b", "user_c"]
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does AI work?",
        "What is deep learning?",
        "Describe transformers",
    ]

    for i in range(45):
        user = random.choice(users)
        query = random.choice(queries)

        result = system.process(user, query)

        status = "OK" if result["success"] else result.get("error", "ERROR")
        cache = "HIT" if result.get("cache_hit") else ""
        latency = result.get("latency_ms", 0)
        level = result.get("degradation_level", "")
        retries = result.get("retries", 0)
        ctx = result.get("context_tokens", 0)
        # Show the context size the LLM actually received (blank on cache hits,
        # which never re-assembled). Watch it shrink when a request degrades.
        ctx_tag = (
            f"ctx={ctx:3d}tok"
            if result["success"] and not result.get("cache_hit")
            else ""
        )
        retry_tag = f"retries={retries}" if retries else ""

        print(
            f"[{i+1:2d}] {user}: {status:15s} {cache:4s} "
            f"{latency:6.0f}ms {level:8s} {ctx_tag:11s} {retry_tag}"
        )

        # Small delay between requests
        time.sleep(0.05)

    # Dashboard
    print("\n" + "-" * 50)
    print("SYSTEM DASHBOARD")
    print("-" * 50)

    dashboard = system.get_dashboard()

    print(f"\n** Health: {dashboard['health']} **\n")

    print("Stats:")
    for key, value in dashboard["stats"].items():
        print(f"  {key}: {value}")

    print(f"\nCache:")
    for key, value in dashboard["cache"].items():
        print(f"  {key}: {value}")

    print(f"\nCosts:")
    costs = dashboard["costs"]
    print(f"  Total: ${costs.get('total_cost', 0):.4f}")
    print(f"  Requests: {costs.get('requests', 0)}")

    print(f"\nCircuit Breaker: {dashboard['circuit']}")

    # Summary
    print("\n" + "-" * 50)
    print("PRODUCTION PATTERNS APPLIED")
    print("-" * 50)
    print("""
✓ Rate Limiting (per-user and global)
✓ Multi-layer Caching (L1 + L2)
✓ Cost Tracking and Budgets
✓ Graceful Degradation
✓ Retry with Exponential Backoff
✓ Circuit Breaker
✓ Real-time Dashboard

This is production-ready code!
""")
