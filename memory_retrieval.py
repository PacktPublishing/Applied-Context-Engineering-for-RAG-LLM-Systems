"""
MEMORY RETRIEVAL
When to pull what memories

Run: python memory_retrieval.py
"""

from dataclasses import dataclass
from typing import List, Dict
import math

# =============================================================================
# TIME DECAY
# One definition of "recent", used everywhere. A memory's weight halves
# every HALF_LIFE_DAYS days (exponential decay).
# =============================================================================

HALF_LIFE_DAYS = 30
DECAY_RATE = math.log(2) / HALF_LIFE_DAYS  # ln(2)/30 ≈ 0.023


def recency_score(age_days: int) -> float:
    """Recency in [0, 1] from a memory's age, via exponential decay."""
    return math.exp(-DECAY_RATE * age_days)


@dataclass
class Memory:
    content: str
    category: str
    importance: float = 0.5
    age_days: int = 0  # days since this memory was created / last relevant
    access_count: int = 0

    @property
    def recency(self) -> float:
        """Derived recency score — newer memories score closer to 1.0."""
        return recency_score(self.age_days)


# =============================================================================
# RETRIEVAL STRATEGIES
# =============================================================================


class AlwaysIncludeRetriever:
    """Some memories should always be in context."""

    def __init__(self, memories: List[Memory]):
        self.memories = memories

    def retrieve(self) -> List[Memory]:
        """Get memories that should always be included."""
        always = []

        for mem in self.memories:
            # Include if it meets ANY always-include rule (added once):
            #   - identity (name, etc.)
            #   - a core preference
            #   - very high importance
            if (
                mem.category == "identity"
                or (mem.category == "preference" and mem.importance >= 0.8)
                or mem.importance >= 0.95
            ):
                always.append(mem)

        return always


class QueryBasedRetriever:
    """Retrieve memories relevant to the current query."""

    def __init__(self, memories: List[Memory]):
        self.memories = memories

    def retrieve(self, query: str, top_k: int = 5) -> List[Memory]:
        """Get memories that actually share keywords with the query."""
        query_words = set(query.lower().split())

        scored = []
        for mem in self.memories:
            mem_words = set(mem.content.lower().split())

            # Keyword overlap (use embeddings in production)
            overlap = len(query_words & mem_words)
            if overlap == 0:
                continue  # no keyword relevance — skip it

            relevance = overlap / max(len(query_words), 1)

            # Combined score
            score = relevance * 0.5 + mem.importance * 0.3 + mem.recency * 0.2

            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]


class CategoryBasedRetriever:
    """Retrieve memories based on intent category."""

    def __init__(self, memories: List[Memory]):
        self.by_category: Dict[str, List[Memory]] = {}
        for mem in memories:
            if mem.category not in self.by_category:
                self.by_category[mem.category] = []
            self.by_category[mem.category].append(mem)

    def retrieve(self, intent: str, per_category: int = 2) -> List[Memory]:
        """Get memories based on detected intent."""
        # Map intents to needed categories
        intent_map = {
            "purchase": ["preference", "financial", "past_purchase"],
            "support": ["identity", "past_issue", "preference"],
            "recommendation": ["preference", "goal", "context"],
            "general": ["identity", "preference"],
        }

        needed = intent_map.get(intent, ["identity", "preference"])

        results = []
        for cat in needed:
            if cat in self.by_category:
                # Get top memories by importance
                sorted_mems = sorted(
                    self.by_category[cat], key=lambda m: m.importance, reverse=True
                )
                results.extend(sorted_mems[:per_category])

        return results


class TimeDecayRetriever:
    """Query relevance + importance + exponential time decay + frequency."""

    def __init__(self, memories: List[Memory]):
        self.memories = memories

    def calculate_score(self, memory: Memory, query_relevance: float) -> float:
        """
        Combined score with time decay.
        Recent memories are weighted higher via exponential decay.
        """
        # Exponential time decay (weight halves every HALF_LIFE_DAYS days)
        time_decay = recency_score(memory.age_days)

        # Frequency boost (frequently accessed = important)
        frequency_factor = min(memory.access_count / 10, 1.0)

        score = (
            query_relevance * 0.4
            + memory.importance * 0.3
            + time_decay * 0.2
            + frequency_factor * 0.1
        )

        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Memory]:
        """Retrieve with time-weighted scoring."""
        query_words = set(query.lower().split())

        scored = []
        for mem in self.memories:
            mem_words = set(mem.content.lower().split())
            overlap = len(query_words & mem_words)
            relevance = overlap / max(len(query_words), 1)

            score = self.calculate_score(mem, relevance)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]


# =============================================================================
# COMPLETE RETRIEVAL PIPELINE
# =============================================================================


class MemoryRetrievalPipeline:
    """
    Complete retrieval pipeline:
    1. Always include core memories
    2. Add query-relevant memories (with time decay)
    3. Add intent-relevant memories
    4. Deduplicate and limit by budget
    """

    def __init__(self, memories: List[Memory]):
        self.memories = memories
        self.always_retriever = AlwaysIncludeRetriever(memories)
        self.time_retriever = TimeDecayRetriever(memories)
        self.category_retriever = CategoryBasedRetriever(memories)

    def retrieve(
        self, query: str, intent: str = "general", max_tokens: int = 500
    ) -> List[Memory]:
        """Full retrieval pipeline."""
        results = []

        # 1. Always include
        always = self.always_retriever.retrieve()
        results.extend(always)

        # 2. Query-relevant (ranked with time decay + frequency)
        query_relevant = self.time_retriever.retrieve(query, top_k=5)
        results.extend(query_relevant)

        # 3. Intent-relevant
        intent_relevant = self.category_retriever.retrieve(intent, per_category=2)
        results.extend(intent_relevant)

        # 4. Deduplicate
        seen = set()
        unique = []
        for mem in results:
            if mem.content not in seen:
                seen.add(mem.content)
                unique.append(mem)

        # 5. Limit by token budget
        total_tokens = 0
        final = []
        for mem in unique:
            mem_tokens = len(mem.content.split()) * 1.3
            if total_tokens + mem_tokens > max_tokens:
                break
            final.append(mem)
            total_tokens += mem_tokens

        return final

    def format_for_context(self, memories: List[Memory]) -> str:
        """Format retrieved memories for LLM context."""
        if not memories:
            return ""

        by_category: Dict[str, List[str]] = {}
        for mem in memories:
            if mem.category not in by_category:
                by_category[mem.category] = []
            by_category[mem.category].append(mem.content)

        lines = ["## Retrieved Memories"]
        for cat, items in by_category.items():
            lines.append(f"\n### {cat.title()}")
            for item in items:
                lines.append(f"- {item}")

        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY RETRIEVAL STRATEGIES")
    print("=" * 60)

    # Sample memories (age_days drives the exponential time decay)
    memories = [
        Memory(
            "User's name is Sarah",
            "identity",
            importance=0.95,
            age_days=50,
            access_count=12,
        ),
        Memory(
            "Works as a data scientist",
            "profession",
            importance=0.8,
            age_days=50,
            access_count=6,
        ),
        Memory(
            "Prefers MacBooks over Windows",
            "preference",
            importance=0.85,
            age_days=30,
            access_count=8,
        ),
        Memory(
            "Budget is around $2000",
            "financial",
            importance=0.9,
            age_days=10,
            access_count=3,
        ),
        Memory(
            "Needs laptop for machine learning",
            "goal",
            importance=0.75,
            age_days=3,
            access_count=2,
        ),
        Memory("Travels frequently for work", "context", importance=0.6, age_days=15),
        Memory(
            "Had issues with keyboard on last laptop",
            "past_issue",
            importance=0.65,
            age_days=70,
        ),
        Memory(
            "Interested in 16GB RAM minimum",
            "preference",
            importance=0.7,
            age_days=6,
            access_count=4,
        ),
        Memory(
            "Uses Python and PyTorch daily",
            "context",
            importance=0.5,
            age_days=20,
            access_count=5,
        ),
        Memory(
            "Prefers lightweight laptops",
            "preference",
            importance=0.8,
            age_days=40,
            access_count=2,
        ),
    ]

    # ---- Strategy spotlight: each strategy on its own ----
    print("\n" + "-" * 40)
    print("STRATEGY SPOTLIGHT")
    print("-" * 40)

    query = "What laptop should I recommend for machine learning?"

    always = AlwaysIncludeRetriever(memories).retrieve()
    print(f"\n[1] Always include -> {[m.content for m in always]}")

    query_hits = QueryBasedRetriever(memories).retrieve(query, top_k=3)
    print(f"[2] Query-based    -> {[m.content for m in query_hits]}")

    cat_hits = CategoryBasedRetriever(memories).retrieve("recommendation")
    print(f"[3] Category-based -> {[m.content for m in cat_hits]}")

    decay_hits = TimeDecayRetriever(memories).retrieve(query, top_k=3)
    print(f"[4] Time-decay     -> {[m.content for m in decay_hits]}")

    # ---- Full pipeline ----
    pipeline = MemoryRetrievalPipeline(memories)
    intent = "recommendation"

    print("\n" + "=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)
    print(f'\nQuery: "{query}"')
    print(f"Intent: {intent}")

    retrieved = pipeline.retrieve(query, intent, max_tokens=300)

    print(f"\nRetrieved {len(retrieved)} memories:")
    for mem in retrieved:
        print(
            f"  [{mem.category}] (imp: {mem.importance}, recency: {mem.recency:.2f}) {mem.content}"
        )

    print("\n" + "-" * 40)
    print("FORMATTED CONTEXT:")
    print("-" * 40)
    print(pipeline.format_for_context(retrieved))

    # Test different intent
    print("\n" + "=" * 60)
    print("DIFFERENT INTENT: support")
    print("=" * 60)

    query2 = "I'm having issues with my laptop"
    retrieved2 = pipeline.retrieve(query2, intent="support", max_tokens=300)

    print(f"\nRetrieved {len(retrieved2)} memories:")
    for mem in retrieved2:
        print(f"  [{mem.category}] {mem.content}")
