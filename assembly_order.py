"""
CONTEXT ASSEMBLY ORDER
Demonstrate why order matters and show the optimal structure

Run: python assembly_order.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
import time
from typing import List

from dotenv import load_dotenv

load_dotenv()
model = "claude-opus-4-5"

client = anthropic.Anthropic()


def compare_assembly_orders():
    """Compare bad vs good context assembly order"""

    print("=" * 60)
    print("CONTEXT ASSEMBLY ORDER COMPARISON")
    print("=" * 60)

    # Simulated content
    question = "What's the refund policy for software purchases?"

    documentation = """
    RETURN POLICY:
    - Hardware: 30-day return window
    - Software: 7-day return if not activated
    - Services: Non-refundable

    SHIPPING POLICY:
    - Standard: 5-7 days
    - Express: 2-3 days
    """ + "Additional policy details... " * 50  # Add some bulk

    # BAD ORDER: Question first, instructions buried
    bad_order = f"""
Question from customer: {question}

Here is our company documentation:
{documentation}

You are a customer service agent. Be helpful and concise.

Please respond in a structured format.
"""

    # GOOD ORDER: Instructions first, question last
    good_order = f"""
## Role
You are a customer service agent for TechCorp. Be helpful, accurate, and concise.

## Relevant Policy Information
{documentation}

## Output Format
Respond with:
1. Direct answer to the question
2. Any relevant caveats
3. Suggested next steps if applicable

## Customer Question
{question}
"""

    # Test both
    print("\n--- BAD ORDER (question first, instructions buried) ---")
    print(f"Context length: {len(bad_order)} chars")

    start = time.time()
    bad_response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": bad_order}],
    )
    bad_time = time.time() - start

    print(f"Response: {bad_response.content[0].text[:200]}...")

    print("\n--- GOOD ORDER (instructions first, question last) ---")
    print(f"Context length: {len(good_order)} chars")

    start = time.time()
    good_response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": good_order}],
    )
    good_time = time.time() - start

    print(f"Response: {good_response.content[0].text[:200]}...")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"\nBad order tokens: {bad_response.usage.input_tokens}")
    print(f"Good order tokens: {good_response.usage.input_tokens}")
    print(f"\nBad order latency: {bad_time:.2f}s")
    print(f"Good order latency: {good_time:.2f}s")


def demonstrate_optimal_order():
    """Show the full optimal assembly order"""

    print("\n" + "=" * 60)
    print("OPTIMAL ASSEMBLY ORDER TEMPLATE")
    print("=" * 60)

    # Build each section
    sections = {
        "1. INSTRUCTIONAL": """## Role
You are a senior technical support engineer.
Be precise, helpful, and cite sources when possible.

## Rules
- Never guess if you don't know
- Always verify information before responding
- Keep responses under 200 words""",
        "2. EXAMPLES": """## Examples

Q: How do I update my profile?
A: Go to Settings > Profile > Edit. Click Save when done.

Q: What's the API rate limit?
A: Free tier: 100/min. Paid: 1000/min. See docs for details.""",
        "3. KNOWLEDGE": """## Retrieved Documentation

[API Reference]
Rate limits apply per API key, not per user.
Exceeded limits return 429 status code.

[Troubleshooting Guide]
If you hit rate limits, implement exponential backoff.""",
        "4. MEMORY": """## Conversation History
User: I'm having API issues
Assistant: What error are you seeing?
User: Getting 429 errors""",
        "5. TOOLS": """## Available Tools
- search_docs(query): Search documentation
- create_ticket(issue): Open support ticket
- get_usage(api_key): Check current usage""",
        "6. OUTPUT": """## Output Format
Respond with:
- Direct answer
- Code example if relevant
- Link to documentation""",
        "7. TASK": """## Current Question
Why am I getting 429 errors and how do I fix it?""",
    }

    full_context = "\n\n---\n\n".join(sections.values())

    print(f"\nFull context ({len(full_context)} chars):\n")

    for name, content in sections.items():
        print(f"\n[{name}]")
        print(content[:150] + "..." if len(content) > 150 else content)


def assembly_order_function():
    """Production-ready assembly function"""

    def assemble_context(
        role: str,
        rules: List[str],
        examples: List[dict],
        retrieved_docs: List[str],
        history: List[dict],
        tools: List[dict],
        output_format: str,
        task: str,
    ) -> str:
        """
        Assemble context in optimal order.

        Order:
        1. Instructional (role, rules)
        2. Examples
        3. Knowledge (retrieved docs)
        4. Memory (history)
        5. Tools
        6. Output format
        7. Task (LAST!)
        """

        parts = []

        # 1. Instructional
        parts.append(f"## Role\n{role}")
        if rules:
            rules_text = "\n".join(f"- {r}" for r in rules)
            parts.append(f"## Rules\n{rules_text}")

        # 2. Examples
        if examples:
            ex_text = "\n\n".join(
                f"Q: {e['question']}\nA: {e['answer']}" for e in examples
            )
            parts.append(f"## Examples\n{ex_text}")

        # 3. Knowledge
        if retrieved_docs:
            docs_text = "\n\n".join(retrieved_docs)
            parts.append(f"## Relevant Information\n{docs_text}")

        # 4. Memory
        if history:
            hist_text = "\n".join(
                f"{h['role']}: {h['content']}" for h in history[-5:]  # Last 5 turns
            )
            parts.append(f"## Conversation History\n{hist_text}")

        # 5. Tools
        if tools:
            tools_text = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
            parts.append(f"## Available Tools\n{tools_text}")

        # 6. Output
        if output_format:
            parts.append(f"## Output Format\n{output_format}")

        # 7. Task (ALWAYS LAST)
        parts.append(f"## Your Task\n{task}")

        return "\n\n".join(parts)

    # Demo the function
    context = assemble_context(
        role="You are a helpful coding assistant.",
        rules=["Be concise", "Include code examples"],
        examples=[
            {"question": "How to read a file?", "answer": "Use open() with 'r' mode."}
        ],
        retrieved_docs=["Python file I/O supports context managers."],
        history=[
            {"role": "User", "content": "I'm working on file processing"},
        ],
        tools=[{"name": "run_code", "description": "Execute Python code"}],
        output_format="Provide code example with explanation.",
        task="How do I read a CSV file efficiently?",
    )

    print("\n" + "=" * 60)
    print("PRODUCTION ASSEMBLY FUNCTION OUTPUT")
    print("=" * 60)
    print(context)


if __name__ == "__main__":
    compare_assembly_orders()
    demonstrate_optimal_order()
    assembly_order_function()
