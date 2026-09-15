"""
CONTEXT ASSEMBLY PIPELINE
Complete production-ready pipeline for context engineering

Run: python context_assembly_pipeline.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time

from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-5"

INPUT_PRICE_PER_MTOK = 3.00
OUTPUT_PRICE_PER_MTOK = 15.00


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Document:
    content: str
    source: str
    category: str


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class UserContext:
    user_id: str
    name: str
    tier: str


@dataclass
class AssembledContext:
    full_text: str
    token_estimate: int
    components_used: List[str]


@dataclass
class PipelineResult:
    answer: str
    context_tokens: int
    response_tokens: int
    components_used: List[str]
    cost_usd: float
    latency_ms: float
    assembled_context: Optional[str] = None


# =============================================================================
# KNOWLEDGE RETRIEVER
# =============================================================================


class KnowledgeRetriever:
    """Simple keyword-based retriever. Replace with embeddings in production."""

    def __init__(self, documents: List[Document]):
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        query_lower = query.lower()

        scored = []
        for doc in self.documents:
            score = sum(
                1 for word in query_lower.split() if word in doc.content.lower()
            )
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


# =============================================================================
# CONTEXT ASSEMBLER
# =============================================================================


class ContextAssembler:
    """Assembles context in optimal order following the framework."""

    def __init__(self, retriever: KnowledgeRetriever, max_tokens: int = 4000):
        self.retriever = retriever
        self.max_tokens = max_tokens

    def assemble(
        self,
        query: str,
        user: Optional[UserContext] = None,
        history: Optional[List[ConversationTurn]] = None,
        examples: Optional[List[Dict]] = None,
    ) -> AssembledContext:
        """Assemble context in optimal order.

        Order: Instructional → Examples → Knowledge → Memory → Output → Task
        (Task LAST — that's the highest-attention position.)
        """

        components: List[str] = []

        # 1. INSTRUCTIONAL (beginning - high attention)
        instruction = self._build_instructional(user)
        components.append("instructional")

        # 2. EXAMPLES (early - sets patterns)
        examples_text = self._build_examples(examples) if examples else None
        if examples_text:
            components.append("examples")

        # 3. KNOWLEDGE (middle - retrieved content)
        docs = self.retriever.retrieve(query)
        knowledge = self._build_knowledge(docs) if docs else None
        if knowledge:
            components.append("knowledge")

        # 4. MEMORY (middle-late - conversation context)
        memory = self._build_memory(history) if history else None
        if memory:
            components.append("memory")

        # 5. OUTPUT (end - format requirements)
        output = self._build_output()
        components.append("output")

        # 6. TASK (very end - highest attention)
        task = f"## Customer Question\n{query}"
        components.append("task")

        middle = [p for p in (examples_text, knowledge, memory) if p]
        full_text = self._join_with_budget(
            head=[instruction],
            middle=middle,
            tail=[output, task],
        )
        token_estimate = len(full_text) // 4

        return AssembledContext(
            full_text=full_text,
            token_estimate=token_estimate,
            components_used=components,
        )

    def _build_instructional(self, user: Optional[UserContext]) -> str:
        tier_note = ""
        if user:
            if user.tier == "premium":
                tier_note = f"\nThis is {user.name}, a PREMIUM customer. Prioritize their satisfaction."
            elif user.tier == "enterprise":
                tier_note = f"\nThis is {user.name}, an ENTERPRISE customer. Provide detailed, thorough responses."

        return f"""## Role
You are a helpful customer support agent for TechCorp.
Be accurate, concise, and friendly.{tier_note}

## Guidelines
- Answer based only on the knowledge provided
- If information isn't available, say so honestly
- Always verify you've fully addressed the question
- Cite sources when referencing specific policies"""

    def _build_examples(self, examples: List[Dict]) -> str:
        formatted = []
        for ex in examples:
            formatted.append(f"Q: {ex['question']}\nA: {ex['answer']}")
        return "## Examples\n\n" + "\n\n".join(formatted)

    def _build_knowledge(self, docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            formatted.append(f"**[{doc.source}]**\n{doc.content}")
        return "## Relevant Information\n\n" + "\n\n".join(formatted)

    def _build_memory(self, history: List[ConversationTurn]) -> str:
        recent = history[-5:]
        formatted = [f"{t.role.capitalize()}: {t.content}" for t in recent]
        return "## Recent Conversation\n" + "\n".join(formatted)

    def _build_output(self) -> str:
        return """## Response Format
Provide your response with:
1. Direct answer to the question
2. Source citations where applicable
3. Offer to help with follow-up questions"""

    def _join_with_budget(
        self,
        head: List[str],
        middle: List[str],
        tail: List[str],
    ) -> str:
        """Join sections, trimming MIDDLE first to preserve head (role) + tail (task).

        Why middle-first: instructions and the task are the highest-attention
        positions in the prompt. Knowledge/memory/examples can be partially
        dropped without breaking the model's ability to follow the request.
        """
        max_chars = self.max_tokens * 4
        sep = "\n\n"

        full = sep.join(head + middle + tail)
        if len(full) <= max_chars:
            return full

        head_text = sep.join(head)
        tail_text = sep.join(tail)
        reserved = len(head_text) + len(tail_text) + (2 * len(sep))
        middle_budget = max_chars - reserved

        if middle_budget <= 0:
            return head_text + sep + tail_text

        kept: List[str] = []
        used = 0
        for section in middle:
            cost = len(section) + len(sep)
            if used + cost > middle_budget:
                break
            kept.append(section)
            used += cost

        trim_note = (
            "\n\n[Context trimmed to fit budget — lower-priority sections dropped]"
        )
        return sep.join(head + kept + tail) + trim_note


# =============================================================================
# SUPPORT PIPELINE
# =============================================================================


class SupportPipeline:
    """Complete support pipeline with context assembly and generation."""

    def __init__(self, documents: List[Document]):
        self.client = anthropic.Anthropic()
        self.retriever = KnowledgeRetriever(documents)
        self.assembler = ContextAssembler(self.retriever)

        # Default few-shot examples
        self.default_examples = [
            {
                "question": "How do I return an item?",
                "answer": "You can return most items within 30 days. Go to your Orders page, select the item, and click 'Request Return'. [Source: Return Policy]",
            },
            {
                "question": "What's the shipping time?",
                "answer": "Standard shipping is 5-7 business days. Express shipping (2-3 days) is available for $12.99. [Source: Shipping Info]",
            },
        ]

    def handle_query(
        self,
        query: str,
        user: Optional[UserContext] = None,
        history: Optional[List[ConversationTurn]] = None,
        use_examples: bool = True,
        return_context: bool = False,
    ) -> PipelineResult:
        """Handle a customer query end-to-end.

        Set return_context=True to also receive the assembled prompt for
        inspection — useful for teaching and debugging.
        """

        start_time = time.time()

        # 1. Assemble context
        context = self.assembler.assemble(
            query=query,
            user=user,
            history=history,
            examples=self.default_examples if use_examples else None,
        )

        # 2. Generate response (with basic error handling)
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": context.full_text}],
            )
        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

        latency_ms = (time.time() - start_time) * 1000

        # 3. Calculate cost
        input_cost = (response.usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_MTOK
        output_cost = (response.usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MTOK

        return PipelineResult(
            answer=response.content[0].text,
            context_tokens=response.usage.input_tokens,
            response_tokens=response.usage.output_tokens,
            components_used=context.components_used,
            cost_usd=input_cost + output_cost,
            latency_ms=latency_ms,
            assembled_context=context.full_text if return_context else None,
        )


# =============================================================================
# DEMO
# =============================================================================


def run_demo():
    """Run a complete demo of the pipeline."""

    print("=" * 70)
    print("CONTEXT ASSEMBLY PIPELINE DEMO")
    print("=" * 70)

    # Setup knowledge base
    documents = [
        Document(
            "Return Policy: Most items can be returned within 30 days of purchase for a full refund. "
            "Electronics have a 15-day return window. Software is non-refundable once activated.",
            "Return Policy Guide",
            "returns",
        ),
        Document(
            "Shipping Options: Standard shipping (5-7 business days) is $5.99. "
            "Express shipping (2-3 business days) is $12.99. "
            "Free shipping on orders over $50.",
            "Shipping Information",
            "shipping",
        ),
        Document(
            "Laptop Warranty: All laptops include a 1-year manufacturer warranty. "
            "Extended warranty (3 years) available for $149. "
            "Covers hardware defects, not accidental damage.",
            "Warranty Policy",
            "warranty",
        ),
        Document(
            "Premium Membership: Premium members get free express shipping, "
            "extended 45-day returns, and priority customer support.",
            "Premium Benefits",
            "membership",
        ),
    ]

    pipeline = SupportPipeline(documents)

    # Test scenarios
    test_cases = [
        {
            "query": "What's the return policy for a laptop I bought?",
            "user": UserContext("u123", "Sarah", "premium"),
            "history": [
                ConversationTurn("user", "Hi, I need help with a purchase"),
                ConversationTurn(
                    "assistant",
                    "Hello! I'd be happy to help. What can I assist you with?",
                ),
            ],
            "show_context": True,  # Inspect the assembled prompt on the first run
        },
        {
            "query": "How long does shipping take?",
            "user": UserContext("u456", "John", "free"),
            "history": None,
            "show_context": False,
        },
        {
            "query": "Is there a warranty on laptops?",
            "user": None,
            "history": None,
            "show_context": False,
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}")
        print(f"{'='*70}")

        print(f"\nQuery: {test['query']}")
        if test["user"]:
            print(f"User: {test['user'].name} ({test['user'].tier})")

        result = pipeline.handle_query(
            query=test["query"],
            user=test["user"],
            history=test["history"],
            return_context=test["show_context"],
        )

        if result.assembled_context:
            print(f"\n--- ASSEMBLED CONTEXT (what the model actually sees) ---")
            print(result.assembled_context)
            print(f"--- END CONTEXT ({len(result.assembled_context)} chars) ---")

        print(f"\n--- ANSWER ---")
        print(result.answer)

        print(f"\n--- METRICS ---")
        print(f"Context tokens: {result.context_tokens:,}")
        print(f"Response tokens: {result.response_tokens:,}")
        print(f"Cost: ${result.cost_usd:.4f}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print(f"Components: {', '.join(result.components_used)}")


if __name__ == "__main__":
    run_demo()
