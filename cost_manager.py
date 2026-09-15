"""
COST MANAGEMENT
Track, budget, and optimize LLM costs

Run: python cost_manager.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# =============================================================================
# COST ENTRY
# =============================================================================


@dataclass
class CostEntry:
    timestamp: datetime
    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    user_id: Optional[str] = None


# =============================================================================
# COST TRACKER
# =============================================================================


class CostTracker:
    """Track and analyze LLM costs."""

    # Illustrative pricing per 1M tokens — always confirm current numbers
    # at https://www.anthropic.com/pricing
    PRICING = {
        "claude-sonnet-5": {"input": 3.00, "output": 15.00},
        "claude-opus-4-8": {"input": 15.00, "output": 75.00},
        "claude-haiku-4-5": {"input": 0.25, "output": 1.25},
    }

    def __init__(self):
        self.entries: List[CostEntry] = []
        self.budgets: Dict[str, float] = {}
        self.alerts: List[Dict] = []

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for a request."""
        pricing = self.PRICING.get(model, {"input": 3.00, "output": 15.00})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def record(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str = None,
    ) -> float:
        """Record a cost entry."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        entry = CostEntry(
            timestamp=datetime.now(),
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            user_id=user_id,
        )

        self.entries.append(entry)
        self._check_budgets(entry)

        return cost

    def _check_budgets(self, entry: CostEntry):
        """Check if any budgets are exceeded."""
        if entry.user_id:
            user_key = f"user:{entry.user_id}"
            if user_key in self.budgets:
                user_total = self.get_user_cost(entry.user_id, period="day")
                if user_total > self.budgets[user_key]:
                    self.alerts.append(
                        {
                            "type": "budget_exceeded",
                            "key": user_key,
                            "budget": self.budgets[user_key],
                            "actual": user_total,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

    def set_budget(self, key: str, daily_budget: float):
        """Set a daily budget."""
        self.budgets[key] = daily_budget

    def get_total_cost(self, period: str = "day") -> float:
        """Get total cost for period."""
        cutoff = self._get_cutoff(period)
        return sum(e.cost for e in self.entries if e.timestamp > cutoff)

    def get_user_cost(self, user_id: str, period: str = "day") -> float:
        """Get cost for specific user."""
        cutoff = self._get_cutoff(period)
        return sum(
            e.cost
            for e in self.entries
            if e.timestamp > cutoff and e.user_id == user_id
        )

    def get_cost_by_operation(self, period: str = "day") -> Dict[str, float]:
        """Get cost breakdown by operation."""
        cutoff = self._get_cutoff(period)
        costs = {}
        for e in self.entries:
            if e.timestamp > cutoff:
                costs[e.operation] = costs.get(e.operation, 0) + e.cost
        return costs

    def get_cost_by_model(self, period: str = "day") -> Dict[str, float]:
        """Get cost breakdown by model."""
        cutoff = self._get_cutoff(period)
        costs = {}
        for e in self.entries:
            if e.timestamp > cutoff:
                costs[e.model] = costs.get(e.model, 0) + e.cost
        return costs

    def _get_cutoff(self, period: str) -> datetime:
        """Get cutoff datetime for period."""
        now = datetime.now()
        if period == "hour":
            return now - timedelta(hours=1)
        elif period == "day":
            return now - timedelta(days=1)
        elif period == "week":
            return now - timedelta(weeks=1)
        elif period == "month":
            return now - timedelta(days=30)
        return now - timedelta(days=1)

    def get_summary(self, period: str = "day") -> Dict:
        """Get cost summary."""
        cutoff = self._get_cutoff(period)
        period_entries = [e for e in self.entries if e.timestamp > cutoff]

        if not period_entries:
            return {"total_cost": 0, "requests": 0}

        return {
            "total_cost": sum(e.cost for e in period_entries),
            "requests": len(period_entries),
            "avg_cost_per_request": sum(e.cost for e in period_entries)
            / len(period_entries),
            "total_input_tokens": sum(e.input_tokens for e in period_entries),
            "total_output_tokens": sum(e.output_tokens for e in period_entries),
            "by_operation": self.get_cost_by_operation(period),
            "by_model": self.get_cost_by_model(period),
        }


# =============================================================================
# BUDGET ENFORCER
# =============================================================================


class BudgetEnforcer:
    """Enforce cost budgets."""

    def __init__(self, tracker: CostTracker):
        self.tracker = tracker
        self.thresholds = {"warning": 0.8, "critical": 0.95, "hard_limit": 1.0}

    def check_budget(self, budget_key: str, period: str = "day") -> Dict:
        """Check budget status."""
        budget = self.tracker.budgets.get(budget_key)
        if not budget:
            return {"status": "no_budget", "allowed": True}

        if budget_key.startswith("user:"):
            user_id = budget_key.replace("user:", "")
            current = self.tracker.get_user_cost(user_id, period)
        else:
            current = self.tracker.get_total_cost(period)

        ratio = current / budget

        if ratio >= self.thresholds["hard_limit"]:
            return {"status": "exceeded", "allowed": False, "usage": ratio}

        if ratio >= self.thresholds["critical"]:
            return {"status": "critical", "allowed": True, "usage": ratio}

        if ratio >= self.thresholds["warning"]:
            return {"status": "warning", "allowed": True, "usage": ratio}

        return {"status": "ok", "allowed": True, "usage": ratio}


# =============================================================================
# COST OPTIMIZER
# =============================================================================


class CostOptimizer:
    """Suggest cost optimizations."""

    def __init__(self, tracker: CostTracker):
        self.tracker = tracker

    def suggest_optimizations(self) -> List[Dict]:
        """Analyze and suggest optimizations."""
        suggestions = []
        summary = self.tracker.get_summary("week")

        if not summary.get("requests"):
            return []

        # Check for model downgrades
        by_model = summary.get("by_model", {})
        if "claude-opus-4-8" in by_model:
            opus_cost = by_model["claude-opus-4-8"]
            if opus_cost > 10:
                suggestions.append(
                    {
                        "type": "model_downgrade",
                        "message": f"Consider Sonnet for some Opus tasks. Opus cost: ${opus_cost:.2f}",
                        "potential_savings": opus_cost * 0.5,
                    }
                )

        # Check for high per-request cost
        avg_cost = summary.get("avg_cost_per_request", 0)
        if avg_cost > 0.05:
            suggestions.append(
                {
                    "type": "caching",
                    "message": f"High per-request cost (${avg_cost:.3f}). Implement caching.",
                    "potential_savings": summary["total_cost"] * 0.3,
                }
            )

        # Check context size
        total_input = summary.get("total_input_tokens", 0)
        requests = summary.get("requests", 1)
        avg_input = total_input / requests
        if avg_input > 5000:
            suggestions.append(
                {
                    "type": "context_reduction",
                    "message": f"Average input is {avg_input:.0f} tokens. Reduce context.",
                    "potential_savings": summary["total_cost"] * 0.2,
                }
            )

        return suggestions


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COST MANAGEMENT")
    print("=" * 60)

    tracker = CostTracker()
    enforcer = BudgetEnforcer(tracker)
    optimizer = CostOptimizer(tracker)

    # Set budgets (tight on purpose so the demo trips an alert)
    tracker.set_budget("global", 2.5)
    tracker.set_budget("user:demo_user", 1.5)

    # Simulate usage
    print("\n" + "-" * 50)
    print("SIMULATING USAGE")
    print("-" * 50)

    models = [
        ("claude-sonnet-5", 5000, 1000),
        ("claude-opus-4-8", 10000, 2000),
        ("claude-haiku-4-5", 3000, 500),
    ]

    for i in range(20):
        model, input_tokens, output_tokens = models[i % 3]
        cost = tracker.record(
            operation="query",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user_id="demo_user",
        )
        print(f"Request {i+1}: {model.split('-')[1]} - ${cost:.4f}")

    # Summary
    print("\n" + "-" * 50)
    print("COST SUMMARY")
    print("-" * 50)

    summary = tracker.get_summary("day")
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print(f"Requests: {summary['requests']}")
    print(f"Avg per request: ${summary['avg_cost_per_request']:.4f}")
    print(f"\nBy model:")
    for model, cost in summary["by_model"].items():
        model_name = model.split("-")[1]
        print(f"  {model_name}: ${cost:.4f}")

    # Budget check
    print("\n" + "-" * 50)
    print("BUDGET STATUS")
    print("-" * 50)

    for key in tracker.budgets:
        check = enforcer.check_budget(key)
        allowed = "allowed" if check["allowed"] else "BLOCKED"
        print(f"{key}: {check['status']} ({check['usage']:.0%} used) -> {allowed}")

    # Alerts fired during recording
    print("\n" + "-" * 50)
    print("BUDGET ALERTS")
    print("-" * 50)
    if tracker.alerts:
        print(f"{len(tracker.alerts)} alert(s) fired")
        latest = tracker.alerts[-1]
        print(
            f"  Latest: {latest['key']} over ${latest['budget']:.2f} budget "
            f"(actual ${latest['actual']:.4f})"
        )
    else:
        print("No alerts fired")

    # Optimizations
    print("\n" + "-" * 50)
    print("OPTIMIZATION SUGGESTIONS")
    print("-" * 50)

    suggestions = optimizer.suggest_optimizations()
    for s in suggestions:
        print(f"\n[{s['type']}]")
        print(f"  {s['message']}")
        print(f"  Potential savings: ${s['potential_savings']:.2f}")

    # Pricing info
    print("\n" + "-" * 50)
    print("PRICING REFERENCE")
    print("-" * 50)
    print("""
| Model  | Input (per 1M) | Output (per 1M) |
|--------|----------------|-----------------|
| Opus   | $15.00         | $75.00          |
| Sonnet | $3.00          | $15.00          |
| Haiku  | $0.25          | $1.25           |
""")
