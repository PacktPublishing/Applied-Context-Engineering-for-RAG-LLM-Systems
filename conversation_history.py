"""
CONVERSATION HISTORY MANAGEMENT
Four strategies for keeping conversations inside a token budget:
sliding window, summarization, selective pruning, and hybrid.

Why manage history at all? On modern models a 15-20k token history
does NOT overflow the context window (they hold 200k+). You manage it
to control COST, LATENCY, and ATTENTION -- the more low-signal history
you send, the more you pay, the slower you respond, and the harder it is
for the model to find what matters.

Run: python conversation_history.py
(Strategy 2 calls Claude if ANTHROPIC_API_KEY is set; otherwise it falls
 back to a clearly-labeled offline stub so the demo still runs.)
"""

import os
from dataclasses import dataclass, field
from typing import List
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# MODEL = "claude-sonnet-4-5"


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


# Current, cheap model for summarization. Update as new models ship.
SUMMARY_MODEL = "claude-haiku-4-5-20251001"


def llm_summarize(text: str) -> str:
    """
    Summarize old conversation turns with a real LLM call.

    This is the actual technique. If no API key is present we fall back to a
    labeled stub so the demo still runs offline -- but note the fallback is
    NOT summarization, it's truncation, which is exactly what you should not
    ship. The point of summarization is that the LLM compresses meaning.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        preview = "; ".join(line[:40] for line in text.splitlines()[:3])
        return f"[OFFLINE STUB - not a real summary] {preview}"

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this conversation history in 2-3 sentences. "
                    "Focus on: key topics, decisions made, and user preferences.\n\n"
                    f"Conversation:\n{text}\n\nSummary:"
                ),
            }
        ],
    )
    return response.content[0].text


# =============================================================================
# STRATEGY 1: SLIDING WINDOW
# =============================================================================


class SlidingWindowHistory:
    """Keep only the last N turns."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Message] = []

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        # Keep only last N turns (1 turn = user + assistant)
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_context(self) -> List[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def token_estimate(self) -> int:
        return sum(len(m.content) for m in self.messages) // 4


# =============================================================================
# STRATEGY 2: SUMMARIZATION
# =============================================================================


class SummarizingHistory:
    """Compress old messages into an LLM-generated summary to save tokens."""

    def __init__(self, recent_turns: int = 5, summary_threshold: int = 15):
        self.recent_turns = recent_turns
        self.summary_threshold = summary_threshold
        self.messages: List[Message] = []
        self.summary: str = ""

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

        # When we exceed threshold, summarize old messages
        if len(self.messages) > self.summary_threshold * 2:
            self._summarize_old()

    def _summarize_old(self):
        """Summarize everything except the most recent turns."""
        recent_count = self.recent_turns * 2
        old_messages = self.messages[:-recent_count]

        old_text = "\n".join(f"{m.role}: {m.content}" for m in old_messages)
        new_summary = llm_summarize(old_text)

        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"
        else:
            self.summary = new_summary

        self.messages = self.messages[-recent_count:]

    def get_context(self) -> List[dict]:
        context = []

        # Inject the summary as a leading user turn. (Anthropic's `system`
        # prompt is a top-level parameter, NOT a role inside messages, so we
        # keep summaries in the user/assistant stream here.)
        if self.summary:
            context.append(
                {
                    "role": "user",
                    "content": f"[Previous conversation summary: {self.summary}]",
                }
            )
            context.append(
                {
                    "role": "assistant",
                    "content": "Understood. I'll keep that context in mind.",
                }
            )

        for m in self.messages:
            context.append({"role": m.role, "content": m.content})

        return context


# =============================================================================
# STRATEGY 3: SELECTIVE PRUNING
# =============================================================================


class SelectivePruningHistory:
    """Score messages by importance, drop low-value ones when over budget."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.messages: List[dict] = []

    def add(self, role: str, content: str):
        importance = self._score_importance(content)
        self.messages.append(
            {
                "role": role,
                "content": content,
                "importance": importance,
            }
        )
        self._prune_if_needed()

    def _score_importance(self, content: str) -> float:
        """Score message importance with simple heuristics (0.1 - 1.0)."""
        score = 0.5

        # Higher importance
        if "?" in content:
            score += 0.1
        if any(w in content.lower() for w in ["important", "remember", "must"]):
            score += 0.2
        if any(w in content.lower() for w in ["my name", "i prefer", "budget"]):
            score += 0.3

        # Lower importance
        if len(content) < 20:
            score -= 0.2
        if content.lower().strip() in ["ok", "thanks", "got it", "sure", "yes", "no"]:
            score -= 0.3

        return max(0.1, min(1.0, score))

    def _prune_if_needed(self):
        """Remove lowest-importance messages until back under budget."""
        total_tokens = sum(len(m["content"]) // 4 for m in self.messages)

        while total_tokens > self.max_tokens and len(self.messages) > 4:
            # Only consider pruning messages outside the most recent 4
            prunable = self.messages[:-4]
            if not prunable:
                break

            min_idx = min(range(len(prunable)), key=lambda i: prunable[i]["importance"])
            removed = self.messages.pop(min_idx)
            total_tokens -= len(removed["content"]) // 4

    def get_context(self) -> List[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]


# =============================================================================
# STRATEGY 4: HYBRID (BEST PRACTICE)
# =============================================================================


class HybridHistory:
    """Summarize old turns, but always keep 'pinned' important messages."""

    def __init__(self, max_recent: int = 10):
        self.max_recent = max_recent
        self.summary: str = ""
        self.pinned: List[Message] = []
        self.recent: List[Message] = []

    def add(self, role: str, content: str, pin: bool = False):
        msg = Message(role=role, content=content)

        if pin or self._should_auto_pin(content):
            self.pinned.append(msg)

        self.recent.append(msg)
        if len(self.recent) > self.max_recent * 2:
            self._rotate()

    def _should_auto_pin(self, content: str) -> bool:
        """Auto-detect messages worth pinning."""
        indicators = ["my name is", "i prefer", "important:", "remember", "budget"]
        return any(ind in content.lower() for ind in indicators)

    def _rotate(self):
        """Summarize the oldest turns, keep the most recent."""
        keep = self.max_recent * 2
        old = self.recent[:-keep]
        self.recent = self.recent[-keep:]

        if old:
            old_text = "\n".join(f"{m.role}: {m.content}" for m in old)
            new_summary = llm_summarize(old_text)
            self.summary = f"{self.summary} {new_summary}".strip()

    def get_context(self) -> List[dict]:
        context = []

        # Summary goes in the user/assistant stream (see note in Strategy 2).
        if self.summary:
            context.append(
                {
                    "role": "user",
                    "content": f"[Conversation summary: {self.summary}]",
                }
            )
            context.append(
                {
                    "role": "assistant",
                    "content": "Understood.",
                }
            )

        # Pinned messages -- but skip any that are still in `recent` so we
        # don't send the same content twice.
        recent_ids = {id(m) for m in self.recent}
        for m in self.pinned:
            if id(m) not in recent_ids:
                context.append({"role": m.role, "content": f"[Pinned] {m.content}"})

        for m in self.recent:
            context.append({"role": m.role, "content": m.content})

        return context


# =============================================================================
# DEMO
# =============================================================================


def simulate_long_conversation() -> List[tuple]:
    """Generate a sample long conversation."""
    return [
        ("user", "Hi! My name is John and I work in marketing."),
        ("assistant", "Hello John! Nice to meet you. How can I help you today?"),
        ("user", "I'm looking for a new CRM system for my team."),
        ("assistant", "I'd be happy to help you find a CRM. What's your team size?"),
        ("user", "We have about 15 people. Budget is around $500/month."),
        ("assistant", "Great! For 15 users at $500/month, I'd recommend..."),
        ("user", "That sounds interesting. What about integrations?"),
        ("assistant", "Most modern CRMs integrate with common tools..."),
        ("user", "We use Slack heavily."),
        (
            "assistant",
            "Perfect, Slack integration is available in all options I mentioned.",
        ),
        ("user", "What about mobile apps?"),
        ("assistant", "All three have mobile apps for iOS and Android."),
        ("user", "Thanks, that's helpful."),
        ("assistant", "You're welcome! Any other questions?"),
        ("user", "Actually, I forgot to mention we need GDPR compliance."),
        ("assistant", "Good point! All enterprise CRMs offer GDPR compliance..."),
        ("user", "Great. Let me discuss with my team."),
        ("assistant", "Sure! Feel free to come back with any questions."),
        ("user", "One more thing - can these handle multiple languages?"),
        ("assistant", "Yes, most support multiple languages..."),
        ("user", "Perfect. I'll be back next week."),
        ("assistant", "Sounds good! Have a great week, John."),
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("CONVERSATION HISTORY STRATEGIES COMPARISON")
    print("=" * 60)

    conversation = simulate_long_conversation()
    print(f"Original conversation: {len(conversation)} messages")

    # Budgets are deliberately small so every strategy actually fires on
    # this short demo conversation.
    strategies = [
        ("Sliding Window (5 turns)", SlidingWindowHistory(max_turns=5)),
        (
            "Summarizing (keep 3)",
            SummarizingHistory(recent_turns=3, summary_threshold=8),
        ),
        ("Selective Pruning", SelectivePruningHistory(max_tokens=150)),
        ("Hybrid", HybridHistory(max_recent=5)),
    ]

    for name, strategy in strategies:
        for role, content in conversation:
            strategy.add(role, content)

        context = strategy.get_context()
        total_chars = sum(len(m["content"]) for m in context)

        print(f"\n{'-' * 40}")
        print(f"{name}")
        print(f"{'-' * 40}")
        print(f"Messages in context: {len(context)}")
        print(f"Characters: {total_chars} (~{total_chars // 4} tokens)")
        print(f"\nFirst message: {context[0]['content'][:60]}...")
        print(f"Last message: {context[-1]['content'][:60]}...")
