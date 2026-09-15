"""
QUALITY VS QUANTITY DEMONSTRATION
Prove that focused context outperforms unfocused context

Run: python quality_vs_quantity.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
import time
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

model = "claude-haiku-4-5-20251001"


def test_context(context: str, question: str, label: str) -> Dict[str, Any]:
    """Test a context/question pair and measure results"""

    start_time = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""Context:
{context}

Question: {question}

Provide a direct, concise answer based only on the context.""",
            }
        ],
    )

    elapsed = time.time() - start_time

    return {
        "label": label,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": elapsed * 1000,
        "answer": response.content[0].text,
        "cost": (response.usage.input_tokens * 0.000003)
        + (response.usage.output_tokens * 0.000015),
    }


def generate_noise(sentences: int = 50) -> str:
    """Generate irrelevant filler content"""
    noise_templates = [
        "The company was established in the early 2000s.",
        "Our headquarters are located in San Francisco.",
        "We have a diverse team of professionals.",
        "Customer satisfaction is our top priority.",
        "We operate in multiple regions globally.",
        "Our technology stack is modern and scalable.",
        "The board of directors meets quarterly.",
        "We sponsor various community events.",
        "Our office features an open floor plan.",
        "Employee wellness programs are available.",
    ]

    return " ".join(noise_templates[i % len(noise_templates)] for i in range(sentences))


def generate_conflicting_noise(sentences: int = 50) -> str:
    """Generate plausible-but-WRONG info about the SAME topic.

    Mimics real-world bad RAG: retrieving stale docs, old pricing pages,
    outdated policies, or competitor terms that look authoritative.
    """
    conflicting_statements = [
        # Wrong pricing
        "The Premium tier is priced at $99 per month for business customers.",
        "Our Basic plan provides 5,000 API calls per month.",
        "Premium subscribers receive 25,000 API calls per billing cycle.",
        # Wrong refund window
        "Refund requests must be submitted within 14 days of the original purchase.",
        "All sales become final after the 7-day evaluation period ends.",
        "We offer a generous 60-day money-back guarantee on annual subscriptions.",
        # Wrong overage rules
        "All overage charges are fully refundable upon written request.",
        "Premium plan overages are billed at a flat $0.10 per additional call.",
        "There are no overage fees on any of our subscription plans.",
        # Wrong policy details
        "Customers exceeding plan limits are automatically upgraded to the next tier.",
        "API rate limits reset every 24 hours regardless of subscription plan.",
        "Pro-rated refunds apply exclusively to the Enterprise tier customers.",
    ]
    return " ".join(
        conflicting_statements[i % len(conflicting_statements)]
        for i in range(sentences)
    )


def run_quality_comparison():
    """Compare focused vs unfocused context"""

    print("=" * 60)
    print("QUALITY VS QUANTITY COMPARISON")
    print("=" * 60)

    # The key information
    relevant_info = """
    PRICING INFORMATION:
    - Basic Plan: $19/month - 1,000 API calls/month
    - Premium Plan: $49/month - 10,000 API calls/month
    - Enterprise Plan: Custom pricing - Unlimited API calls

    OVERAGE FEES (charged when monthly limit exceeded):
    - Basic Plan: $0.05 per additional API call
    - Premium Plan: $0.02 per additional API call
    - Enterprise Plan: No overage fees

    REFUND POLICY:
    - 30-day money-back guarantee: full refund within first 30 days
    - After 30 days: pro-rated refund for unused portion of current month only
    - Overage fees are NEVER refundable
    """

    # Multi-hop synthesis question:
    # Premium = 10,000 included; 12,000 used = 2,000 overage @ $0.02 = $40
    # Cancel at day 45 = past 30 days = pro-rated refund of unused days in current month
    # Day 45 = 15 days into month 2 = 15 unused days of $49 = ~$24.50 refund
    # Overage of $40 is NOT refundable
    question = (
        "A customer on the Premium plan made 12,000 API calls last month "
        "and now wants to cancel on day 45 of their subscription. "
        "What overage fees did they incur, and what refund are they entitled to? "
        "Show your math."
    )

    # Create different context scenarios
    scenarios = {
        "Minimal (just relevant)": relevant_info,
        "Focused (relevant + structure)": f"""
## Pricing & Policies

{relevant_info}

Use this information to answer customer questions about our plans.
""",
        "Moderate noise": f"""
{generate_noise(20)}

{relevant_info}

{generate_noise(20)}
""",
        "Heavy noise": f"""
{generate_noise(100)}

{relevant_info}

{generate_noise(100)}
""",
        "Extreme noise": f"""
{generate_noise(300)}

{relevant_info}

{generate_noise(300)}
""",
        "Conflicting noise (bad RAG)": f"""
{generate_conflicting_noise(20)}

{relevant_info}

{generate_conflicting_noise(20)}
""",
    }

    results = []

    for label, context in scenarios.items():
        print(f"\nTesting: {label}...", end=" ", flush=True)
        result = test_context(context, question, label)
        results.append(result)
        print(f"Done ({result['input_tokens']} tokens)")

    # Display comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    print(f"\n{'Scenario':<25} {'Tokens':>8} {'Latency':>10} {'Cost':>10}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['label']:<25} {r['input_tokens']:>8,} {r['latency_ms']:>8.0f}ms ${r['cost']:>8.4f}"
        )

    # Calculate relevance density
    print("\n" + "=" * 60)
    print("RELEVANCE DENSITY")
    print("=" * 60)

    relevant_tokens = results[0]["input_tokens"]  # Baseline

    print(f"\nRelevant content: ~{relevant_tokens} tokens\n")

    for r in results:
        density = (relevant_tokens / r["input_tokens"]) * 100
        bar = "█" * int(density / 5)
        print(f"{r['label']:<25} {density:>5.1f}% {bar}")

    # Show answers
    print("\n" + "=" * 60)
    print("ANSWER QUALITY (check for accuracy)")
    print("=" * 60)

    # Expected correct answer:
    # - Overage: 2,000 calls × $0.02 = $40 (NOT refundable)
    # - Refund: pro-rated for 15 unused days of $49/month ≈ $24.50
    print("\nEXPECTED: Overage = $40 (2,000 × $0.02, non-refundable)")
    print("EXPECTED: Refund ≈ $24.50 (pro-rated, 15/30 days × $49)\n")

    for r in results:
        print(f"\n{'-' * 60}")
        print(f"{r['label']}:")
        print(f"{'-' * 60}")
        print(r["answer"])


def calculate_relevance_density(context: str, relevant_content: str) -> float:
    """Calculate relevance density for a context"""
    # Simple approximation: character ratio
    return len(relevant_content) / len(context)


if __name__ == "__main__":
    run_quality_comparison()
