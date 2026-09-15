"""
LOST IN THE MIDDLE DEMONSTRATION
Reproduce the finding from the "Lost in the Middle" research using:
  - An older model (claude-3-haiku-20240307) where the U-shape effect still appears
  - Competing needles: many customer records with the same shape but different values,
    so the model must locate the SPECIFIC record by ID, not just any policy fact.

Run: python lost_in_the_middle.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
import random
import time
from typing import Literal

from dotenv import load_dotenv

load_dotenv()
model = "claude-haiku-4-5"
client = anthropic.Anthropic()

Position = Literal["beginning", "middle", "end"]

TARGET_ID = "A4242"
TARGET_WINDOW = 45  # days


def generate_records(num_records: int, exclude_id: str, seed: int = 42) -> list[str]:
    """Generate competing customer records — same shape, different values."""
    rng = random.Random(seed)
    possible_windows = [7, 14, 21, 30, 60, 90, 120, 180]
    records: list[str] = []
    used = {exclude_id}
    while len(records) < num_records:
        candidate = f"A{rng.randint(1, 9999):04d}"
        if candidate in used:
            continue
        used.add(candidate)
        window = rng.choice(possible_windows)
        records.append(f"Customer {candidate}: refund window {window} days.")
    return records


def test_retrieval_by_position(
    position: Position,
    num_records: int = 1000,
) -> dict:
    """Test if the model retrieves a specific customer record based on its position."""

    records = generate_records(num_records, exclude_id=TARGET_ID)
    target_record = f"Customer {TARGET_ID}: refund window {TARGET_WINDOW} days."

    if position == "beginning":
        records.insert(0, target_record)
    elif position == "end":
        records.append(target_record)
    else:  # middle
        records.insert(len(records) // 2, target_record)

    context = "\n".join(records)
    question = f"What is customer {TARGET_ID}'s refund window in days?"

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"""Below is a database of customer refund records, one per line.

{context}

Question: {question}

Answer with just the number of days based only on the records above.""",
            }
        ],
    )
    elapsed = time.time() - start_time
    answer = response.content[0].text

    # Strict check: must contain the exact target number
    return {
        "position": position,
        "context_tokens": response.usage.input_tokens,
        "answer": answer,
        "latency_ms": elapsed * 1000,
        "found_fact": str(TARGET_WINDOW) in answer,
    }


def run_lost_in_middle_experiment():
    """Reproduce the Lost in the Middle finding with the competing-needles pattern."""
    print("=" * 60)
    print("LOST IN THE MIDDLE EXPERIMENT")
    print(f"Model: {model}")
    print("=" * 60)

    num_records = 1000
    print(f"\nTarget: Customer {TARGET_ID} (refund window {TARGET_WINDOW} days)")
    print(f"Distractors: {num_records} other customers with windows of 7-180 days")
    print("\nTesting retrieval by position...\n")

    results = []
    for position in ["beginning", "middle", "end"]:
        print(f"Testing {position}...", end=" ", flush=True)
        result = test_retrieval_by_position(position, num_records=num_records)
        results.append(result)
        status = "✓ FOUND" if result["found_fact"] else "✗ MISSED"
        print(f"{status}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for r in results:
        print(f"\n{r['position'].upper()}:")
        print(f"  Context tokens: {r['context_tokens']:,}")
        print(f"  Latency: {r['latency_ms']:.0f}ms")
        print(f"  Found fact: {'Yes' if r['found_fact'] else 'NO'}")
        print(f"  Answer: {r['answer'][:100]}...")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    found = sum(1 for r in results if r["found_fact"])
    print(f"\nFacts found: {found}/3")

    if not results[1]["found_fact"]:
        print("\n⚠️  LOST IN THE MIDDLE EFFECT OBSERVED")
        print("   The record buried in the middle was harder to retrieve.")
    else:
        print("\n✓ Model performed well across all positions")
        print("  (Try increasing num_records or using a weaker model.)")


def test_with_increasing_context():
    """Test how retrieval degrades with increasing context size at middle position."""
    print("\n" + "=" * 60)
    print("CONTEXT SIZE VS RETRIEVAL ACCURACY")
    print("=" * 60)

    sizes = [200, 600, 1500, 3000]
    print(f"\nTarget: Customer {TARGET_ID} buried at middle position")
    print("Testing with increasing distractor count...\n")

    for size in sizes:
        result = test_retrieval_by_position("middle", num_records=size)
        status = "✓" if result["found_fact"] else "✗"
        print(
            f"Records: {size:>4} | Tokens: {result['context_tokens']:>6,} | Found: {status}"
        )


if __name__ == "__main__":
    run_lost_in_middle_experiment()
    test_with_increasing_context()
