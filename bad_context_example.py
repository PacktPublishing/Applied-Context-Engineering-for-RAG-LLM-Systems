"""
BAD CONTEXT ENGINEERING EXAMPLE
Demonstrates what NOT to do: dumping entire documents into context

Run: python bad_context_example.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
import time
from dotenv import load_dotenv

load_dotenv()
model = "claude-opus-4-5"

client = anthropic.Anthropic()

# Simulating a large documentation dump (in reality, this would be 50K+ tokens)
ENTIRE_DOCUMENTATION = """
# Product Documentation v4.2.1

## Chapter 1: Getting Started
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...

## Chapter 2: Installation
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu
fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident...

## Chapter 3: Configuration
[... imagine 50 more pages of documentation ...]

## Chapter 47: API Reference
### Authentication
To authenticate, use your API key in the Authorization header:
Authorization: Bearer YOUR_API_KEY

Rate limits: 100 requests per minute for free tier, 1000 for paid.

## Chapter 48: Troubleshooting
[... more content ...]
""" * 20  # Multiply to simulate large document

USER_QUESTION = "What are the API rate limits?"


def bad_context_approach():
    """
    BAD: Dump everything into context
    Problems:
    - Expensive (paying for 50K+ tokens)
    - Slow (model has to process everything)
    - Inaccurate (important info gets "lost in the middle")
    """
    print("=" * 60)
    print("BAD CONTEXT ENGINEERING")
    print("=" * 60)

    # Count tokens (approximate)
    total_chars = len(ENTIRE_DOCUMENTATION) + len(USER_QUESTION)
    approx_tokens = total_chars // 4
    print(f"Approximate tokens in context: {approx_tokens:,}")

    start_time = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""Here is our entire product documentation:

{ENTIRE_DOCUMENTATION}

User question: {USER_QUESTION}

Please answer the user's question based on the documentation.""",
            }
        ],
    )

    elapsed = time.time() - start_time

    print(f"Response time: {elapsed:.2f} seconds")
    print(f"Answer: {response.content[0].text}")
    print(f"\nInput tokens: {response.usage.input_tokens:,}")
    print(f"Cost estimate: ${response.usage.input_tokens * 0.000003:.4f}")

    return response


if __name__ == "__main__":
    bad_context_approach()
