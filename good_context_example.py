"""
GOOD CONTEXT ENGINEERING EXAMPLE
Demonstrates the right approach: retrieve only relevant information

Run: python good_context_example.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
import time
from typing import List
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

model = "claude-opus-4-5"

# Simulated document chunks (in reality, these would be in a vector database)
DOCUMENT_CHUNKS = [
    {
        "id": "auth-001",
        "content": "To authenticate with the API, use your API key in the Authorization header: Authorization: Bearer YOUR_API_KEY",
        "section": "Authentication",
    },
    {
        "id": "rate-001",
        "content": "API Rate Limits: Free tier allows 100 requests per minute. Paid tier allows 1,000 requests per minute. Enterprise tier has custom limits.",
        "section": "Rate Limits",
    },
    {
        "id": "install-001",
        "content": "To install the SDK, run: pip install our-sdk. Requires Python 3.8 or higher.",
        "section": "Installation",
    },
    {
        "id": "errors-001",
        "content": "Common error codes: 401 = Invalid API key, 429 = Rate limit exceeded, 500 = Server error",
        "section": "Error Handling",
    },
    # ... imagine hundreds more chunks
]


def simple_retrieval(query: str, top_k: int = 3) -> List[dict]:
    """
    Simple keyword-based retrieval (in production, use embeddings + vector search)
    """
    query_lower = query.lower()
    scored_chunks = []

    for chunk in DOCUMENT_CHUNKS:
        # Simple scoring: count keyword matches
        score = 0
        for word in query_lower.split():
            if word in chunk["content"].lower():
                score += 1
            if word in chunk["section"].lower():
                score += 2  # Section matches are more valuable

        if score > 0:
            scored_chunks.append((score, chunk))

    # Sort by score, return top_k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k]]


USER_QUESTION = "What are the API rate limits?"


def good_context_approach():
    """
    GOOD: Retrieve only relevant information
    Benefits:
    - Cheap (paying for ~1K tokens instead of 50K)
    - Fast (model processes less)
    - Accurate (relevant info is prominent, not buried)
    """
    print("=" * 60)
    print("GOOD CONTEXT ENGINEERING")
    print("=" * 60)

    # Step 1: Retrieve relevant chunks
    relevant_chunks = simple_retrieval(USER_QUESTION, top_k=3)

    print(f"Retrieved {len(relevant_chunks)} relevant chunks:")
    for chunk in relevant_chunks:
        print(f"  - [{chunk['section']}]: {chunk['content'][:50]}...")

    # Step 2: Assemble focused context
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(f"[{chunk['section']}]\n{chunk['content']}")

    assembled_context = "\n\n".join(context_parts)

    # Count tokens (approximate)
    total_chars = len(assembled_context) + len(USER_QUESTION)
    approx_tokens = total_chars // 4
    print(f"\nApproximate tokens in context: {approx_tokens:,}")

    start_time = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""## Relevant Documentation

{assembled_context}

## User Question
{USER_QUESTION}

## Instructions
Answer based ONLY on the documentation above. Be concise and specific.""",
            }
        ],
    )

    elapsed = time.time() - start_time

    print(f"\nResponse time: {elapsed:.2f} seconds")
    print(f"Answer: {response.content[0].text}")
    print(f"\nInput tokens: {response.usage.input_tokens:,}")
    print(f"Cost estimate: ${response.usage.input_tokens * 0.000003:.4f}")

    return response


if __name__ == "__main__":
    good_context_approach()
