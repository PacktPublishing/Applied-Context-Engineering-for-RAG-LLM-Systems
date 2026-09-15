"""
GRACEFUL DEGRADATION
Fallback strategies for context systems

Run: python graceful_degradation.py
"""

from dataclasses import dataclass
from typing import Callable, Any, List, Optional
from enum import Enum

# =============================================================================
# DEGRADATION LEVELS
# =============================================================================


class DegradationLevel(Enum):
    FULL = 1  # Everything works
    REDUCED = 2  # Some features limited
    FALLBACK = 3  # Using backup systems
    MINIMAL = 4  # Basic functionality only
    FAILED = 5  # Nothing works


@dataclass
class DegradationStep:
    level: DegradationLevel
    name: str
    function: Callable
    description: str


# =============================================================================
# GRACEFUL DEGRADER
# =============================================================================


class GracefulDegrader:
    """Manage graceful degradation across service levels."""

    def __init__(self):
        self.steps: List[DegradationStep] = []
        self.current_level = DegradationLevel.FULL
        self.failure_log: List[dict] = []

    def add_step(
        self, level: DegradationLevel, name: str, function: Callable, description: str
    ):
        """Add a degradation step."""
        self.steps.append(DegradationStep(level, name, function, description))
        self.steps.sort(key=lambda s: s.level.value)

    def execute(self, *args, **kwargs) -> tuple[Any, DegradationLevel]:
        """Execute with automatic degradation on failure."""
        for step in self.steps:
            try:
                result = step.function(*args, **kwargs)
                self.current_level = step.level
                return result, step.level
            except Exception as e:
                self.failure_log.append(
                    {"level": step.level.name, "step": step.name, "error": str(e)}
                )
                continue

        self.current_level = DegradationLevel.FAILED
        return None, DegradationLevel.FAILED

    def get_status(self) -> dict:
        """Get current degradation status."""
        return {
            "current_level": self.current_level.name,
            "recent_failures": self.failure_log[-5:],
            "available_steps": [s.name for s in self.steps],
        }


# =============================================================================
# RETRIEVAL DEGRADER
# =============================================================================


class RetrievalDegrader:
    """Graceful degradation for retrieval systems."""

    def __init__(self):
        self.degrader = GracefulDegrader()
        self._setup_steps()

    def _setup_steps(self):
        """Setup degradation levels."""
        self.degrader.add_step(
            DegradationLevel.FULL,
            "semantic_search",
            self._semantic_search,
            "Full vector similarity search",
        )
        self.degrader.add_step(
            DegradationLevel.REDUCED,
            "hybrid_search",
            self._hybrid_search,
            "Combined vector + keyword",
        )
        self.degrader.add_step(
            DegradationLevel.FALLBACK,
            "keyword_search",
            self._keyword_search,
            "Keyword-only search",
        )
        self.degrader.add_step(
            DegradationLevel.MINIMAL,
            "static_results",
            self._static_results,
            "Return generic results",
        )

    def retrieve(self, query: str, k: int = 5) -> tuple[list, DegradationLevel]:
        """Retrieve with graceful degradation."""
        return self.degrader.execute(query, k)

    def _semantic_search(self, query: str, k: int) -> list:
        """Full semantic search - simulate failure sometimes."""
        import random

        if random.random() < 0.3:  # 30% failure rate for demo
            raise Exception("Vector DB connection failed")

        return [
            {"content": f"Semantic result for: {query}", "score": 0.95},
            {"content": "Another semantic match", "score": 0.85},
        ]

    def _hybrid_search(self, query: str, k: int) -> list:
        """Hybrid search - less accurate but more reliable."""
        return [
            {"content": f"Hybrid result for: {query}", "score": 0.80},
            {"content": "Keyword-assisted match", "score": 0.75},
        ]

    def _keyword_search(self, query: str, k: int) -> list:
        """Keyword-only fallback."""
        return [{"content": f"Keyword match for: {query}", "score": 0.60}]

    def _static_results(self, query: str, k: int) -> list:
        """Static/cached fallback."""
        return [{"content": "Generic helpful information", "score": 0.0}]


# =============================================================================
# MODEL DEGRADER
# =============================================================================


class ModelDegrader:
    """Graceful degradation across model tiers."""

    def __init__(self):
        self.degrader = GracefulDegrader()
        self._setup_steps()

    def _setup_steps(self):
        """Setup model degradation levels (most capable -> cheapest)."""
        self.degrader.add_step(
            DegradationLevel.FULL,
            "claude-opus-4-8",
            lambda msgs: self._call_model("claude-opus-4-8", msgs),
            "Most capable model",
        )
        self.degrader.add_step(
            DegradationLevel.REDUCED,
            "claude-sonnet-5",
            lambda msgs: self._call_model("claude-sonnet-5", msgs),
            "Good balance (default)",
        )
        self.degrader.add_step(
            DegradationLevel.FALLBACK,
            "claude-haiku-4-5",
            lambda msgs: self._call_model("claude-haiku-4-5", msgs),
            "Fast and cheap",
        )
        self.degrader.add_step(
            DegradationLevel.MINIMAL,
            "cached",
            self._cached_response,
            "Return cached response",
        )

    def generate(self, messages: list) -> tuple[str, DegradationLevel]:
        """Generate with model degradation."""
        return self.degrader.execute(messages)

    def _call_model(self, model: str, messages: list) -> str:
        """Simulate model call - top tier fails sometimes (overload)."""
        import random

        if model == "claude-opus-4-8" and random.random() < 0.4:
            raise Exception("claude-opus-4-8 overloaded")
        return f"Response from {model}: {messages[-1].get('content', '')[:30]}..."

    def _cached_response(self, messages: list) -> str:
        """Return cached response."""
        return "I'm experiencing high demand. Please try again."


# =============================================================================
# COMPLETE DEGRADING PIPELINE
# =============================================================================


class DegradingPipeline:
    """Complete pipeline with degradation at every stage."""

    def __init__(self):
        self.retrieval = RetrievalDegrader()
        self.model = ModelDegrader()

    def process(self, query: str) -> dict:
        """Process query with graceful degradation."""
        levels = {}

        # Retrieval
        retrieved, retrieval_level = self.retrieval.retrieve(query)
        levels["retrieval"] = retrieval_level.name

        # Generation
        messages = [{"role": "user", "content": f"Context: {retrieved}\n\n{query}"}]
        response, model_level = self.model.generate(messages)
        levels["model"] = model_level.name

        return {
            "response": response,
            "retrieved": retrieved,
            "degradation_levels": levels,
            "fully_degraded": any(
                l in ["FALLBACK", "MINIMAL", "FAILED"] for l in levels.values()
            ),
        }


# =============================================================================
# USER NOTICE
# =============================================================================


def get_degradation_notice(levels: dict) -> Optional[str]:
    """Generate user notice for degraded service."""
    if all(l == "FULL" for l in levels.values()):
        return None

    if any(l == "MINIMAL" for l in levels.values()):
        return "Note: Operating with limited capabilities."

    if any(l == "FALLBACK" for l in levels.values()):
        return "Note: Some features temporarily unavailable."

    return None


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GRACEFUL DEGRADATION")
    print("=" * 60)

    pipeline = DegradingPipeline()

    # Run multiple queries to show degradation
    print("\n" + "-" * 50)
    print("PROCESSING QUERIES (with random failures)")
    print("-" * 50)

    for i in range(10):
        result = pipeline.process(f"Query {i+1}: What is AI?")

        levels = result["degradation_levels"]
        notice = get_degradation_notice(levels)

        print(f"\nQuery {i+1}:")
        print(f"  Retrieval: {levels['retrieval']}")
        print(f"  Model: {levels['model']}")
        print(f"  Response: {result['response'][:50]}...")
        if notice:
            print(f"  Notice: {notice}")

    # Show degrader status
    print("\n" + "-" * 50)
    print("DEGRADER STATUS")
    print("-" * 50)

    print(f"\nRetrieval: {pipeline.retrieval.degrader.get_status()}")
    print(f"Model: {pipeline.model.degrader.get_status()}")

    # Summary
    print("\n" + "-" * 50)
    print("DEGRADATION LEVELS")
    print("-" * 50)
    print("""
| Level    | Meaning                           |
|----------|-----------------------------------|
| FULL     | Everything working perfectly      |
| REDUCED  | Some features limited             |
| FALLBACK | Using backup systems              |
| MINIMAL  | Basic functionality only          |
| FAILED   | Service unavailable               |
""")
