import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

model = "claude-opus-4-5"


def count_tokens(text: str, model: str = model) -> int:
    """Count tokens in a text string using the Anthropic API"""
    response = client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens


def estimate_tokens(text: str) -> int:
    """Quick estimate: ~4 characters per token for English"""
    return len(text) // 4


def demonstrate_tokenization():
    """Show how different texts tokenize differently"""

    examples = [
        "Hello",
        "Hello world",
        "Context engineering",
        "The quick brown fox jumps over the lazy dog.",
        "antidisestablishmentarianism",
        "def calculate_total(items): return sum(i.price for i in items)",
        "你好世界",  # Chinese: "Hello world"
    ]

    print("=" * 60)
    print("TOKEN COUNTING EXAMPLES")
    print("=" * 60)

    for text in examples:
        actual = count_tokens(text)
        estimated = estimate_tokens(text)

        print(f"\nText: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  Characters: {len(text)}")
        print(f"  Actual tokens: {actual}")
        print(f"  Estimated (chars/4): {estimated}")
        print(f"  Ratio: {len(text)/actual:.1f} chars/token")


def context_window_breakdown():
    """Show how context window gets divided"""

    print("\n" + "=" * 60)
    print("CONTEXT WINDOW BREAKDOWN")
    print("=" * 60)

    # Simulate a typical request
    system_prompt = """You are a helpful customer service agent for TechCorp.
    Be concise and accurate. Follow company policies."""

    conversation_history = """
    User: I bought a laptop last week.
    Assistant: I'd be happy to help with your laptop purchase.
    User: It's not turning on.
    Assistant: I'm sorry to hear that. Let me help troubleshoot.
    """

    retrieved_docs = """
    TROUBLESHOOTING GUIDE:
    1. Check power connection
    2. Hold power button for 10 seconds
    3. Try a different outlet
    4. Check battery indicator
    """

    current_question = "I tried all the steps and it still won't turn on. What now?"

    components = {
        "System prompt": system_prompt,
        "Conversation history": conversation_history,
        "Retrieved documents": retrieved_docs,
        "Current question": current_question,
    }

    total = 0
    for name, content in components.items():
        tokens = count_tokens(content)
        total += tokens
        print(f"\n{name}:")
        print(f"  Tokens: {tokens}")

    print(f"\n{'='*40}")
    print(f"TOTAL INPUT TOKENS: {total}")
    print(f"Context window (Claude Sonnet): 200,000")
    print(f"Remaining for response: {200000 - total:,}")
    print(f"Usage: {total/200000*100:.3f}%")


if __name__ == "__main__":
    demonstrate_tokenization()
    context_window_breakdown()
