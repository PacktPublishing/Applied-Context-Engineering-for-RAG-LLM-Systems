"""
PROMPT ENGINEERING VS CONTEXT ENGINEERING
Side-by-side comparison of approaches

Run: python context_comparison.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
model = "claude-opus-4-5"

# =============================================================================
# PROMPT ENGINEERING APPROACH
# Manual, one-off, reactive
# =============================================================================


def prompt_engineering_approach(question: str) -> str:
    """
    Traditional prompt engineering: manually craft each prompt
    """
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""You are a helpful customer service agent for TechCorp.
Be concise but friendly. Always greet the customer first.
If you don't know something, say so honestly.
Our business hours are 9 AM - 5 PM EST, Monday through Friday.
Our return policy is 30 days for most items.

Customer question: {question}""",
            }
        ],
    )
    return response.content[0].text


# =============================================================================
# CONTEXT ENGINEERING APPROACH
# Systematic, automated, proactive
# =============================================================================


@dataclass
class CustomerProfile:
    id: str
    name: str
    tier: str  # "free", "premium", "enterprise"
    account_age_days: int
    recent_purchases: List[str]


@dataclass
class KnowledgeChunk:
    id: str
    content: str
    category: str
    relevance_score: float


# Simulated data stores
CUSTOMERS_DB = {
    "cust_123": CustomerProfile(
        id="cust_123",
        name="Sarah",
        tier="premium",
        account_age_days=450,
        recent_purchases=["Laptop Pro X", "USB-C Hub"],
    )
}

KNOWLEDGE_BASE = [
    KnowledgeChunk(
        "kb_001",
        "Password reset: Go to Settings > Security > Reset Password. You'll receive an email within 5 minutes.",
        "account",
        0.0,
    ),
    KnowledgeChunk(
        "kb_002",
        "Return policy: 30 days for most items, 90 days for premium members.",
        "returns",
        0.0,
    ),
    KnowledgeChunk(
        "kb_003",
        "Premium members get priority support with 2-hour response time.",
        "support",
        0.0,
    ),
    KnowledgeChunk(
        "kb_004",
        "Business hours: 9 AM - 5 PM EST, Monday through Friday. Premium members have 24/7 chat support.",
        "hours",
        0.0,
    ),
]


def get_customer_profile(customer_id: str) -> Optional[CustomerProfile]:
    """Retrieve customer profile from database"""
    return CUSTOMERS_DB.get(customer_id)


def search_knowledge_base(query: str, top_k: int = 3) -> List[KnowledgeChunk]:
    """
    Search knowledge base for relevant chunks
    (In production: use embeddings + vector search)
    """
    query_lower = query.lower()
    scored_chunks = []

    for chunk in KNOWLEDGE_BASE:
        score = 0
        for word in query_lower.split():
            if word in chunk.content.lower():
                score += 1
        chunk.relevance_score = score
        if score > 0:
            scored_chunks.append(chunk)

    scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
    return scored_chunks[:top_k]


def get_conversation_history(customer_id: str, limit: int = 3) -> List[Dict]:
    """Get recent conversation history"""
    # Simulated - in production, fetch from database
    return [
        {"role": "user", "content": "I bought a laptop last week"},
        {
            "role": "assistant",
            "content": "Great! How can I help you with your Laptop Pro X?",
        },
    ]


def assemble_context(
    customer: Optional[CustomerProfile],
    knowledge: List[KnowledgeChunk],
    history: List[Dict],
    question: str,
) -> str:
    """
    Assemble context dynamically based on available information
    This is the CORE of context engineering
    """
    parts = []

    # Customer context (if available)
    if customer:
        parts.append(f"""## Customer Profile
- Name: {customer.name}
- Membership: {customer.tier.upper()}
- Customer since: {customer.account_age_days} days
- Recent purchases: {', '.join(customer.recent_purchases)}""")

    # Relevant knowledge
    if knowledge:
        knowledge_text = "\n".join([f"- {k.content}" for k in knowledge])
        parts.append(f"""## Relevant Information
{knowledge_text}""")

    # Conversation history
    if history:
        history_text = "\n".join(
            [f"- {h['role']}: {h['content']}" for h in history[-3:]]
        )
        parts.append(f"""## Recent Conversation
{history_text}""")

    # Current question
    parts.append(f"""## Current Question
{question}""")

    # Instructions (tailored to customer tier)
    if customer and customer.tier == "premium":
        parts.append("""## Instructions
This is a PREMIUM customer. Prioritize their satisfaction.
Mention their premium benefits when relevant.
Be warm and use their name.""")
    else:
        parts.append("""## Instructions
Be helpful and concise.
If you don't know something, say so honestly.""")

    return "\n\n".join(parts)


def context_engineering_approach(customer_id: str, question: str) -> str:
    """
    Context engineering: build a system that assembles optimal context
    """
    # 1. Retrieve customer context
    customer = get_customer_profile(customer_id)

    # 2. Retrieve relevant knowledge
    knowledge = search_knowledge_base(question)

    # 3. Get conversation history
    history = get_conversation_history(customer_id)

    # 4. Assemble context
    assembled_context = assemble_context(customer, knowledge, history, question)

    # 5. Generate response
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": assembled_context}],
    )

    return response.content[0].text


# =============================================================================
# COMPARISON
# =============================================================================

if __name__ == "__main__":
    question = "How do I reset my password?"

    print("=" * 60)
    print("PROMPT ENGINEERING APPROACH")
    print("=" * 60)
    print(prompt_engineering_approach(question))

    print("\n" + "=" * 60)
    print("CONTEXT ENGINEERING APPROACH")
    print("=" * 60)
    print(context_engineering_approach("cust_123", question))
