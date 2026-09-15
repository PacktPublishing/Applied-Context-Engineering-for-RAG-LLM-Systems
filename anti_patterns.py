"""
THE 5 CONTEXT ENGINEERING ANTI-PATTERNS
Examples of what NOT to do (and how to fix it)

Run: python anti_patterns.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

model = "claude-opus-4-5"


# =============================================================================
# ANTI-PATTERN 1: DOCUMENT DUMP
# =============================================================================


def antipattern_document_dump(question: str, all_docs: str) -> str:
    """
    BAD: Dump entire documentation into context
    Problems: Expensive, slow, attention dilution
    """
    # This might be 50K-100K tokens!
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": f"{all_docs}\n\nQuestion: {question}"}],
    )
    return response.content[0].text


def fix_document_dump(question: str, doc_chunks: List[str]) -> str:
    """
    GOOD: Retrieve only relevant chunks
    """
    # Retrieve top 3 relevant chunks (maybe 1-2K tokens)
    relevant = retrieve_relevant(question, doc_chunks, top_k=3)

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""Relevant information:
{chr(10).join(relevant)}

Question: {question}""",
            }
        ],
    )
    return response.content[0].text


# =============================================================================
# ANTI-PATTERN 2: STATIC CONTEXT
# =============================================================================

STATIC_SYSTEM_PROMPT = """You are a helpful assistant.
Be concise and accurate.
Our business hours are 9-5 EST."""


def antipattern_static_context(question: str) -> str:
    """
    BAD: Same static prompt for every user, every situation
    Problems: Not personalized, not adaptive, misses relevant info
    """
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=STATIC_SYSTEM_PROMPT,  # Same for everyone!
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def fix_static_context(question: str, user_id: str, context_type: str) -> str:
    """
    GOOD: Dynamic context based on user and situation
    """
    # Build context dynamically
    user = get_user_profile(user_id)
    relevant_docs = search_docs(question)

    dynamic_context = f"""## User Context
- Name: {user['name']}
- Tier: {user['tier']}
- Previous issues: {user['recent_issues']}

## Relevant Knowledge
{chr(10).join(relevant_docs)}

## Instructions
{"Be extra helpful - this is a premium customer." if user['tier'] == 'premium' else "Be helpful and concise."}
"""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=dynamic_context,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# =============================================================================
# ANTI-PATTERN 3: LOST IN THE MIDDLE
# =============================================================================


def antipattern_lost_in_middle(critical_info: str, question: str) -> str:
    """
    BAD: Critical information buried in the middle
    Problems: Models have U-shaped attention, middle gets ignored
    """
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""Here is some background information:
{generate_filler_text(2000)}

Important policy information:
{critical_info}

Additional context:
{generate_filler_text(2000)}

Question: {question}""",
            }
        ],
    )
    return response.content[0].text


def fix_lost_in_middle(critical_info: str, question: str) -> str:
    """
    GOOD: Critical information at START or END, clear structure
    """
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""## CRITICAL POLICY (Read First)
{critical_info}

## Question
{question}

## Instructions
Answer based on the CRITICAL POLICY section above.""",
            }
        ],
    )
    return response.content[0].text


# =============================================================================
# ANTI-PATTERN 4: NO BUDGET TRACKING
# =============================================================================


def antipattern_no_budget(context_parts: List[str]) -> str:
    """
    BAD: No awareness of token usage or cost
    Problems: Unpredictable costs, context overflow
    """
    # Just concatenate everything, no limits
    full_context = "\n".join(context_parts)

    # No idea how many tokens this is!
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": full_context}],
    )
    return response.content[0].text


@dataclass
class TokenBudget:
    total: int = 4000
    system: int = 500
    knowledge: int = 2000
    history: int = 1000
    question: int = 500


def fix_with_budget(context_parts: Dict[str, str], budget: TokenBudget) -> str:
    """
    GOOD: Explicit token budgets for each context component
    """
    # Truncate each part to its budget
    system = truncate_to_tokens(context_parts.get("system", ""), budget.system)
    knowledge = truncate_to_tokens(context_parts.get("knowledge", ""), budget.knowledge)
    history = truncate_to_tokens(context_parts.get("history", ""), budget.history)
    question = truncate_to_tokens(context_parts.get("question", ""), budget.question)

    assembled = f"""## System
{system}

## Knowledge
{knowledge}

## History
{history}

## Question
{question}"""

    # Track actual usage
    estimated_tokens = len(assembled) // 4
    print(f"Context tokens: {estimated_tokens} / {budget.total} budget")

    response = client.messages.create(
        model=model, max_tokens=500, messages=[{"role": "user", "content": assembled}]
    )

    # Log actual usage
    print(f"Actual input tokens: {response.usage.input_tokens}")

    return response.content[0].text


# =============================================================================
# ANTI-PATTERN 5: NO MEASUREMENT
# =============================================================================


def antipattern_no_measurement(context: str, question: str) -> str:
    """
    BAD: No logging, no metrics, no way to improve
    Problems: Can't debug, can't iterate, flying blind
    """
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": f"{context}\n\n{question}"}],
    )
    # Response goes out, nothing logged
    return response.content[0].text


@dataclass
class ContextMetrics:
    request_id: str
    context_tokens: int
    response_tokens: int
    retrieval_count: int
    latency_ms: float
    cost_usd: float


def fix_with_measurement(context: str, question: str) -> tuple[str, ContextMetrics]:
    """
    GOOD: Log everything, measure everything
    """
    import time
    import uuid

    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": f"{context}\n\n{question}"}],
    )

    latency_ms = (time.time() - start_time) * 1000

    # Calculate metrics
    metrics = ContextMetrics(
        request_id=request_id,
        context_tokens=response.usage.input_tokens,
        response_tokens=response.usage.output_tokens,
        retrieval_count=context.count("##"),  # Rough proxy
        latency_ms=latency_ms,
        cost_usd=(response.usage.input_tokens * 0.000003)
        + (response.usage.output_tokens * 0.000015),
    )

    # Log metrics (in production: send to observability platform)
    print(
        f"[{metrics.request_id}] Tokens: {metrics.context_tokens} in, {metrics.response_tokens} out"
    )
    print(
        f"[{metrics.request_id}] Latency: {metrics.latency_ms:.0f}ms, Cost: ${metrics.cost_usd:.4f}"
    )

    return response.content[0].text, metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def retrieve_relevant(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Simple keyword-based retrieval (use embeddings in production)"""
    scored = [
        (sum(1 for w in query.lower().split() if w in c.lower()), c) for c in chunks
    ]
    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]


def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Mock user profile lookup"""
    return {
        "name": "Sarah",
        "tier": "premium",
        "recent_issues": ["billing", "shipping"],
    }


def search_docs(query: str) -> List[str]:
    """Mock document search"""
    return [
        "Policy: 30-day returns for all items.",
        "Premium members get free shipping.",
    ]


def generate_filler_text(chars: int) -> str:
    """Generate filler text for demos"""
    return ("Lorem ipsum dolor sit amet. " * (chars // 30))[:chars]


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximate token limit"""
    max_chars = max_tokens * 4  # Rough approximation
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ANTI-PATTERN DEMO")
    print("=" * 60)

    # Demo: No measurement vs With measurement
    context = "Our return policy is 30 days. Premium members get 90 days."
    question = "What's the return policy?"

    print("\n" + "-" * 40)
    print("ANTI-PATTERN 5: NO MEASUREMENT vs FIX")
    print("-" * 40)

    print("\nWithout measurement:")
    result = antipattern_no_measurement(context, question)
    print(f"Response: {result[:100]}...")

    print("\nWith measurement:")
    result, metrics = fix_with_measurement(context, question)
    print(f"Response: {result[:100]}...")
