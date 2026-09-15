"""
CONTEXT AUDIT PROJECT
Compare a poorly-assembled context against an engineered one.

Run: python context_audit_project.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List

import anthropic

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# Sonnet 4.6 pricing (per 1M tokens) — keep in sync with platform.claude.com/docs/en/pricing
MODEL = "claude-sonnet-4-6"
INPUT_PRICE_PER_M = 3.00
OUTPUT_PRICE_PER_M = 15.00

# Daily request volume used for the monthly-savings projection
REQUESTS_PER_DAY = 10_000
DAYS_PER_MONTH = 30


# =============================================================================
# SIMULATED DATA (In production, these come from databases / retrieval systems)
# =============================================================================

COMPANY_HISTORY = """
TechCorp was founded in 2005 by John Smith and Jane Doe. Starting in a small
garage in Silicon Valley, the company grew from 2 employees to over 5,000
worldwide. Our journey has been marked by innovation, perseverance, and a
commitment to customer satisfaction. In 2010, we launched our first product...
""" + "Company milestone details. " * 200

ALL_POLICIES = """
SHIPPING POLICY:
Standard shipping: 5-7 business days, $5.99
Express shipping: 2-3 business days, $12.99
Overnight shipping: Next business day, $24.99

RETURN POLICY:
All items can be returned within 30 days of purchase for a full refund.
Items must be in original packaging and unused condition.
Electronics have a 15-day return window.
Software is non-refundable once activated.

PRIVACY POLICY:
We collect minimal personal data and never sell it to third parties.

WARRANTY POLICY:
All products include a 1-year manufacturer warranty.

PRICE MATCH POLICY:
We match competitor pricing on identical items within 14 days of purchase.
""" + "Policy details continue. " * 100

PRODUCT_CATALOG = """
LAPTOPS:
- TechBook Pro 15" - $1,299 - Intel i7, 16GB RAM, 512GB SSD
- TechBook Air 13" - $999 - Intel i5, 8GB RAM, 256GB SSD
- TechBook Gaming - $1,799 - Intel i9, 32GB RAM, 1TB SSD

PHONES:
- TechPhone X - $899 - 6.5" display, 128GB
- TechPhone SE - $499 - 5.8" display, 64GB

ACCESSORIES:
Cases, chargers, cables, stands, docks, and more.
""" + "Product specifications continue. " * 200

FAQ_FULL = """
Q: How do I track my order?
A: Log into your account and click "Order History".

Q: What payment methods do you accept?
A: We accept Visa, Mastercard, AMEX, PayPal.
""" + "FAQ continues. " * 100

TEAM_BIOS = """
LEADERSHIP TEAM:
John Smith, CEO - 20 years experience in tech.
Jane Doe, CTO - PhD in Computer Science from MIT.
""" + "Bio details. " * 50

RECENT_ANNOUNCEMENTS = """
NEW: Spring sale starts March 1st!
UPDATE: Extended warranty now available.
NOTICE: Holiday shipping deadlines posted.
""" + "Announcement details. " * 50


# =============================================================================
# BAD CONTEXT ASSEMBLY (What we're auditing)
# =============================================================================


def build_bad_context(question: str) -> str:
    """
    BAD: Dumps everything into context.
    - Mostly irrelevant content
    - Customer question buried in the middle
    - No structure or signposting
    """
    return f"""
Here is information about our company:

{COMPANY_HISTORY}

Our policies:
{ALL_POLICIES}

Our products:
{PRODUCT_CATALOG}

Frequently asked questions:
{FAQ_FULL}

Customer question: {question}

Our team:
{TEAM_BIOS}

Recent announcements:
{RECENT_ANNOUNCEMENTS}

Please answer the customer's question.
"""


# =============================================================================
# GOOD CONTEXT ASSEMBLY (The fix)
# =============================================================================


def retrieve_relevant_policy(question: str) -> str:
    q = question.lower()

    if "return" in q or "refund" in q:
        return (
            "RETURN POLICY:\n"
            "All items can be returned within 30 days of purchase for a full refund.\n"
            "Items must be in original packaging and unused condition.\n"
            "Electronics have a 15-day return window.\n"
            "Software is non-refundable once activated."
        )

    if "ship" in q or "deliver" in q:
        return (
            "SHIPPING POLICY:\n"
            "Standard shipping: 5-7 business days, $5.99\n"
            "Express shipping: 2-3 business days, $12.99\n"
            "Overnight shipping: Next business day, $24.99"
        )

    if "warrant" in q:
        return (
            "WARRANTY POLICY:\n"
            "All products include a 1-year manufacturer warranty.\n"
            "Extended warranty available for purchase."
        )

    return (
        "GENERAL POLICY:\n"
        "Contact customer service for specific policy questions.\n"
        "Email: support@techcorp.com | Phone: 1-800-TECH"
    )


def retrieve_relevant_products(question: str) -> str:
    q = question.lower()

    if "laptop" in q:
        return (
            "LAPTOPS:\n"
            '- TechBook Pro 15" - $1,299\n'
            '- TechBook Air 13" - $999\n'
            "- TechBook Gaming - $1,799"
        )

    if "phone" in q:
        return "PHONES:\n" "- TechPhone X - $899\n" "- TechPhone SE - $499"

    return ""


def build_good_context(question: str) -> str:
    """
    GOOD: Engineered context assembly.
    - Only relevant sections retrieved
    - Structured with markdown headers
    - Question placed at the END (recency primacy)
    """
    parts = ["## Relevant Policy", retrieve_relevant_policy(question)]

    products = retrieve_relevant_products(question)
    if products:
        parts.extend(["\n## Related Products", products])

    parts.extend(
        [
            "\n## Customer Question",
            question,
            "\n## Instructions",
            "Answer based on the policy above. Be concise and helpful.",
        ]
    )

    return "\n".join(parts)


# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================


@dataclass
class AuditMetrics:
    label: str
    input_tokens: int
    output_tokens: int
    keyword_recall: float  # fraction of expected keywords present in answer
    latency_ms: float
    cost_usd: float  # input + output cost for THIS call
    answer: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _extract_text(response) -> str:
    """Pull text from the first text block; tolerate other block types."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def evaluate_context(
    client: anthropic.Anthropic,
    context: str,
    expected_keywords: List[str],
    label: str,
) -> AuditMetrics:
    """Send `context` to the model and score the answer."""
    start = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": context}],
    )
    latency_ms = (time.time() - start) * 1000.0

    answer = _extract_text(response)
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    found = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    keyword_recall = found / len(expected_keywords) if expected_keywords else 0.0

    cost_usd = (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_M
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    )

    return AuditMetrics(
        label=label,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        keyword_recall=keyword_recall,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        answer=answer,
    )


def _pct_change(before: float, after: float) -> float:
    """Percent reduction from `before` to `after`. Returns 0 if before is ~0."""
    if before <= 0:
        return 0.0
    return (before - after) / before * 100.0


def print_audit_report(before: AuditMetrics, after: AuditMetrics) -> None:
    print("\n" + "=" * 70)
    print("  CONTEXT AUDIT REPORT")
    print("=" * 70)
    print(f"\n{'Metric':<22} {'Before':>14} {'After':>14} {'Change':>14}")
    print("-" * 70)

    token_reduction = _pct_change(before.total_tokens, after.total_tokens)
    print(
        f"{'Total tokens':<22} {before.total_tokens:>14,} "
        f"{after.total_tokens:>14,} {token_reduction:>13.0f}%"
    )

    print(
        f"{'  Input tokens':<22} {before.input_tokens:>14,} "
        f"{after.input_tokens:>14,} {_pct_change(before.input_tokens, after.input_tokens):>13.0f}%"
    )
    print(
        f"{'  Output tokens':<22} {before.output_tokens:>14,} "
        f"{after.output_tokens:>14,} {_pct_change(before.output_tokens, after.output_tokens):>13.0f}%"
    )

    print(
        f"{'Latency (ms)':<22} {before.latency_ms:>14.0f} "
        f"{after.latency_ms:>14.0f} {_pct_change(before.latency_ms, after.latency_ms):>13.0f}%"
    )

    recall_change = (after.keyword_recall - before.keyword_recall) * 100
    print(
        f"{'Keyword recall':<22} {before.keyword_recall*100:>13.0f}% "
        f"{after.keyword_recall*100:>13.0f}% {recall_change:>+12.0f}pts"
    )

    cost_reduction = _pct_change(before.cost_usd, after.cost_usd)
    print(
        f"{'Cost per call ($)':<22} {before.cost_usd:>14.5f} "
        f"{after.cost_usd:>14.5f} {cost_reduction:>13.0f}%"
    )

    # Monthly projection
    monthly_before = before.cost_usd * REQUESTS_PER_DAY * DAYS_PER_MONTH
    monthly_after = after.cost_usd * REQUESTS_PER_DAY * DAYS_PER_MONTH
    monthly_savings = monthly_before - monthly_after

    print("\n" + "=" * 70)
    print(f"  MONTHLY PROJECTION ({REQUESTS_PER_DAY:,} requests/day)")
    print("=" * 70)
    print(f"  Before:  ${monthly_before:>12,.2f} / month")
    print(f"  After:   ${monthly_after:>12,.2f} / month")
    print(
        f"  SAVINGS: ${monthly_savings:>12,.2f} / month "
        f"(${monthly_savings * 12:,.2f} / year)"
    )


def run_audit() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()

    print("=" * 70)
    print("  CONTEXT AUDIT PROJECT")
    print(f"  Model: {MODEL}")
    print("=" * 70)

    question = "What is your return policy? How many days do I have to return an item?"
    expected_keywords = ["30 days", "refund", "original packaging"]

    print(f"\nTest question: {question}")
    print(f"Expected keywords in answer: {expected_keywords}")

    bad_context = build_bad_context(question)
    good_context = build_good_context(question)

    print(f"\nBad context:  {len(bad_context):>7,} characters")
    print(f"Good context: {len(good_context):>7,} characters")

    print("\nCalling API for both contexts...")
    try:
        before = evaluate_context(
            client, bad_context, expected_keywords, "Before (Bad)"
        )
        after = evaluate_context(
            client, good_context, expected_keywords, "After (Good)"
        )
    except anthropic.AuthenticationError:
        print("ERROR: Invalid ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    except anthropic.RateLimitError as e:
        retry_after = (
            e.response.headers.get("retry-after", "unknown")
            if e.response
            else "unknown"
        )
        print(f"ERROR: Rate limited. Retry after {retry_after}s.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        print(f"ERROR: API returned {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as e:
        print(f"ERROR: Could not reach Anthropic API: {e}", file=sys.stderr)
        return 1

    print_audit_report(before, after)

    print("\n" + "=" * 70)
    print("  ANSWERS")
    print("=" * 70)
    print(f"\nBEFORE (truncated to 300 chars):\n  {before.answer[:300]}")
    print(f"\nAFTER (truncated to 300 chars):\n  {after.answer[:300]}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(run_audit())
