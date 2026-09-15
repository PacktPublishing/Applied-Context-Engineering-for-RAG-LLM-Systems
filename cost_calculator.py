"""
TOKEN COST CALCULATOR
Calculate actual costs for API usage

Run: python cost_calculator.py
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelPricing:
    """Pricing per 1M tokens (as of May 2026)"""
    input_per_million: float
    output_per_million: float
    name: str


# Current pricing (update as needed)
PRICING: Dict[str, ModelPricing] = {
    "claude-sonnet": ModelPricing(3.00, 15.00, "Claude Sonnet 4.6"),
    "claude-opus": ModelPricing(15.00, 75.00, "Claude Opus 4.7"),
    "claude-haiku": ModelPricing(1.00, 5.00, "Claude Haiku 4.5"),
    "gpt-4o": ModelPricing(2.50, 10.00, "GPT-4o"),
    "gpt-4-turbo": ModelPricing(10.00, 30.00, "GPT-4 Turbo"),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50, "GPT-3.5 Turbo"),
}


def calculate_request_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet"
) -> float:
    """Calculate cost for a single request"""
    pricing = PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
    return input_cost + output_cost


def calculate_monthly_cost(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "claude-sonnet"
) -> Dict[str, float]:
    """Calculate monthly cost projection"""
    cost_per_request = calculate_request_cost(avg_input_tokens, avg_output_tokens, model)
    daily_cost = cost_per_request * requests_per_day
    monthly_cost = daily_cost * 30
    yearly_cost = monthly_cost * 12

    return {
        "cost_per_request": cost_per_request,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "yearly_cost": yearly_cost,
    }


def compare_context_strategies():
    """Compare bad vs good context engineering costs"""

    print("=" * 60)
    print("CONTEXT ENGINEERING COST COMPARISON")
    print("=" * 60)

    # Scenario: Customer service bot, 10K requests/day
    requests_per_day = 10_000
    output_tokens = 500  # Same response size

    scenarios = {
        "Bad Context (Document Dump)": {
            "input_tokens": 50_000,
            "description": "Dump entire documentation into context"
        },
        "Medium Context (Some RAG)": {
            "input_tokens": 10_000,
            "description": "Basic retrieval, no optimization"
        },
        "Good Context (Engineered)": {
            "input_tokens": 2_000,
            "description": "Targeted retrieval, structured context"
        },
    }

    results = {}

    for name, config in scenarios.items():
        costs = calculate_monthly_cost(
            requests_per_day,
            config["input_tokens"],
            output_tokens
        )
        results[name] = costs

        print(f"\n{name}")
        print(f"  {config['description']}")
        print(f"  Input tokens: {config['input_tokens']:,}")
        print(f"  Cost per request: ${costs['cost_per_request']:.4f}")
        print(f"  Daily cost: ${costs['daily_cost']:,.2f}")
        print(f"  Monthly cost: ${costs['monthly_cost']:,.2f}")
        print(f"  Yearly cost: ${costs['yearly_cost']:,.2f}")

    # Calculate savings
    bad = results["Bad Context (Document Dump)"]["monthly_cost"]
    good = results["Good Context (Engineered)"]["monthly_cost"]
    savings = bad - good
    savings_pct = (savings / bad) * 100

    print(f"\n{'='*60}")
    print(f"MONTHLY SAVINGS: ${savings:,.2f} ({savings_pct:.0f}%)")
    print(f"YEARLY SAVINGS: ${savings * 12:,.2f}")
    print("=" * 60)


def attention_cost_visualization():
    """Illustrative mental model — NOT a real attention measurement.

    Real transformer attention does not decay as a simple 1/N curve. This is a
    teaching device to show that piling more tokens into context dilutes the
    model's focus on any single token. Numbers are for intuition only.
    """

    print("\n" + "=" * 60)
    print("ATTENTION DILUTION (illustrative model — not measured)")
    print("=" * 60)

    scenarios = [
        (2_000, 95, "Focused context"),
        (10_000, 60, "Moderate context"),
        (50_000, 20, "Bloated context"),
        (100_000, 10, "Massive context"),
    ]

    header = f"{'Token Count':>11} | {'Relevance':>9} | {'Illustrative Share':>18} | {'Quality':<20} Scenario"
    print("\n" + header)
    print("-" * len(header))

    # Scale bars against the max possible quality_score so the widest bar fits in 20 chars
    max_quality = max((r * (100 / (t / 1000))) / 100 for t, r, _ in scenarios)

    for tokens, relevance, desc in scenarios:
        attention_per_token = 100 / (tokens / 1000)
        quality_score = (relevance * attention_per_token) / 100

        bar_width = int((quality_score / max_quality) * 20)
        quality_bar = "█" * bar_width

        print(f"{tokens:>11,} | {relevance:>8}% | {attention_per_token:>17.1f}% | {quality_bar:<20} {desc}")


if __name__ == "__main__":
    compare_context_strategies()
    attention_cost_visualization()