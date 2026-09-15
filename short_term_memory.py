"""
SHORT-TERM MEMORY
Session-scoped fact extraction

Run: python short_term_memory.py
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime


@dataclass
class Fact:
    content: str
    category: str  # preference, constraint, context, state, identity
    confidence: float = 0.5
    extracted_at: datetime = field(default_factory=datetime.now)
    source_turn: int = 0


class ShortTermMemory:
    """
    Session-scoped memory that extracts facts from conversation.
    Facts are categorized and scored for relevance.
    """

    def __init__(self):
        self.facts: List[Fact] = []
        self.current_turn: int = 0

    def extract_facts(self, message: str, turn_number: int) -> List[Fact]:
        """
        Extract facts from a user message.
        In production, use an LLM for this.
        """
        self.current_turn = turn_number
        extracted = []

        # Simple pattern-based extraction (use LLM in production)
        message_lower = message.lower()

        # Preferences
        if "prefer" in message_lower or "like" in message_lower:
            extracted.append(Fact(
                content=message,
                category="preference",
                confidence=0.8,
                source_turn=turn_number
            ))

        # Budget/constraints
        if "$" in message or "budget" in message_lower:
            extracted.append(Fact(
                content=message,
                category="constraint",
                confidence=0.9,
                source_turn=turn_number
            ))

        # Name / identity — require an explicit self-introduction with a
        # capitalized name so we don't mislabel messages like "I'm looking..."
        # (keyword matching is fragile; an LLM extractor avoids this entirely).
        if re.search(r"\bmy name is\s+[A-Z]", message) or \
                re.search(r"\bi'?m\s+[A-Z][a-z]+\b", message):
            extracted.append(Fact(
                content=message,
                category="identity",
                confidence=0.95,
                source_turn=turn_number
            ))

        # Requirements
        if "need" in message_lower or "must have" in message_lower or "require" in message_lower:
            extracted.append(Fact(
                content=message,
                category="constraint",
                confidence=0.85,
                source_turn=turn_number
            ))

        # Context
        if "looking for" in message_lower or "trying to" in message_lower:
            extracted.append(Fact(
                content=message,
                category="context",
                confidence=0.7,
                source_turn=turn_number
            ))

        # Transient state — low-value, decays quickly (dropped after 10 turns)
        if "comparing" in message_lower or " vs " in message_lower \
                or "right now" in message_lower or "currently" in message_lower:
            extracted.append(Fact(
                content=message,
                category="state",
                confidence=0.6,
                source_turn=turn_number
            ))

        self.facts.extend(extracted)
        return extracted

    def get_relevant_facts(self, query: str, max_facts: int = 5) -> List[Fact]:
        """Get facts relevant to a query."""
        query_words = set(query.lower().split())

        scored = []
        for fact in self.facts:
            fact_words = set(fact.content.lower().split())
            overlap = len(query_words & fact_words)
            score = overlap / max(len(query_words), 1) + fact.confidence * 0.3
            scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:max_facts]]

    def cleanup(self):
        """Remove stale or low-value facts."""
        self.facts = [f for f in self.facts if self._should_keep(f)]

    def _should_keep(self, fact: Fact) -> bool:
        """Decide if a fact should be kept."""
        # Always keep preferences and identity
        if fact.category in ["preference", "identity"]:
            return True

        # Always keep constraints
        if fact.category == "constraint":
            return True

        # Drop old context/state
        if fact.category in ["context", "state"]:
            turns_ago = self.current_turn - fact.source_turn
            if turns_ago > 10:
                return False

        return True

    def to_context(self) -> str:
        """Format all facts for injection into context."""
        if not self.facts:
            return ""

        by_category: Dict[str, List[str]] = {}
        for fact in self.facts:
            if fact.category not in by_category:
                by_category[fact.category] = []
            by_category[fact.category].append(fact.content)

        lines = ["## User Context (This Session)"]
        for category, facts in by_category.items():
            lines.append(f"\n### {category.title()}")
            for fact in facts:
                lines.append(f"- {fact}")

        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SHORT-TERM MEMORY EXTRACTION")
    print("=" * 60)

    stm = ShortTermMemory()

    # Simulate conversation
    messages = [
        "Hi, my name is Sarah and I'm a software developer.",
        "I'm looking for a new laptop for coding and video editing.",
        "Right now I'm comparing the MacBook Pro vs the Dell XPS.",
        "My budget is around $1500, maybe up to $2000 for the right machine.",
        "I prefer MacBooks but I'm open to Windows.",
        "I need at least 16GB RAM and a good display.",
        "Thanks for the suggestions!",
        "Actually, I also need Thunderbolt ports for my monitors.",
    ]

    print("\nExtracting facts from conversation...\n")

    for i, msg in enumerate(messages, 1):
        print(f"Turn {i}: \"{msg}\"")
        extracted = stm.extract_facts(msg, turn_number=i)
        if extracted:
            for fact in extracted:
                print(f"  → [{fact.category}] (conf: {fact.confidence})")

    print("\n" + "=" * 60)
    print("ALL EXTRACTED FACTS")
    print("=" * 60)
    print(stm.to_context())

    print("\n" + "=" * 60)
    print("RELEVANT FACTS FOR QUERY")
    print("=" * 60)

    query = "What laptop should I recommend for video editing?"
    print(f"Query: \"{query}\"\n")

    relevant = stm.get_relevant_facts(query, max_facts=3)
    for fact in relevant:
        print(f"  [{fact.category}] {fact.content}")

    print("\n" + "=" * 60)
    print("CLEANUP / DECAY")
    print("=" * 60)
    print(f"Facts before cleanup: {len(stm.facts)}")

    # Jump ahead 15 turns — transient 'state' facts are now stale and dropped,
    # while preferences, constraints, and identity are always kept.
    stm.current_turn = len(messages) + 15
    stm.cleanup()

    print(f"Facts after cleanup (turn {stm.current_turn}): {len(stm.facts)}")
    print("Remaining categories:", sorted({f.category for f in stm.facts}))
