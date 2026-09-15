"""
COMPLETE CHATBOT MEMORY SYSTEM
Full memory-enabled chatbot with persistent, cross-session memory.

This file mirrors lesson 5.7 exactly. It ships TWO memory extractors:
  1. RuleBasedMemoryExtractor  -> deterministic keywords, no API calls (default)
  2. LLMMemoryExtractor         -> Claude decides what's worth remembering

Start with the rule-based version to understand the pipeline, then swap in the
LLM version for real-world flexibility (see the bottom of demo()).

Run: python chatbot_memory_system.py
Requires: ANTHROPIC_API_KEY environment variable
Models (July 2026): chat -> claude-sonnet-5, extraction -> claude-haiku-4-5
"""

import anthropic
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json
import os

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Message:
    role: str
    content: str
    # ISO string so it survives json.dump without a custom encoder
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Memory:
    content: str
    category: str  # identity, preference, financial, profession, requirement, fact
    importance: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserProfile:
    user_id: str
    name: Optional[str] = None
    memories: List[Memory] = field(default_factory=list)
    session_count: int = 0
    last_seen: Optional[str] = None

    def add_memory(self, memory: Memory):
        # Don't store the exact same fact twice
        existing = [m.content for m in self.memories]
        if memory.content not in existing:
            self.memories.append(memory)

    def get_memories_by_category(self, category: str) -> List[Memory]:
        return [m for m in self.memories if m.category == category]


# =============================================================================
# CONVERSATION MANAGER  (sliding window + lightweight summary)
# =============================================================================


class ConversationManager:
    """Keeps the last N turns verbatim and collapses older turns into a short
    summary line. This is a deterministic, zero-cost summary — see the lesson's
    'Under the hood vs. managed' box for how you'd upgrade this to an LLM
    summary or the API's native compaction."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Message] = []
        self.summary: str = ""

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        if len(self.messages) > self.max_turns * 2:
            self._summarize_old()

    def _summarize_old(self):
        keep = self.max_turns
        old = self.messages[:-keep]
        topics = [m.content[:40] for m in old if m.role == "user"][:3]
        if topics:
            self.summary += f" Earlier the user mentioned: {'; '.join(topics)}."
        self.messages = self.messages[-keep:]

    def get_context(self) -> List[dict]:
        context = []
        if self.summary:
            context.append(
                {"role": "user", "content": f"[Earlier context:{self.summary.strip()}]"}
            )
            context.append(
                {
                    "role": "assistant",
                    "content": "Understood — I remember that context.",
                }
            )
        for m in self.messages:
            context.append({"role": m.role, "content": m.content})
        return context


# =============================================================================
# MEMORY STORE  (persistence)
# =============================================================================


class MemoryStore:
    """Persist user profiles to disk (JSON for the demo; use a real DB in prod)."""

    def __init__(self, storage_path: str = "/tmp/chatbot_memory.json"):
        self.storage_path = storage_path
        self.profiles: Dict[str, UserProfile] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            for uid, pdata in data.items():
                # Saved dicts carry exactly the Memory fields, so Memory(**m) works
                memories = [Memory(**m) for m in pdata.get("memories", [])]
                self.profiles[uid] = UserProfile(
                    user_id=uid,
                    name=pdata.get("name"),
                    memories=memories,
                    session_count=pdata.get("session_count", 0),
                    last_seen=pdata.get("last_seen"),
                )
        except Exception:
            # Corrupt store shouldn't crash the bot — start fresh
            pass

    def _save(self):
        data = {}
        for uid, p in self.profiles.items():
            data[uid] = {
                "name": p.name,
                "memories": [
                    {
                        "content": m.content,
                        "category": m.category,
                        "importance": m.importance,
                        "created_at": m.created_at,
                    }
                    for m in p.memories
                ],
                "session_count": p.session_count,
                "last_seen": p.last_seen,
            }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_profile(self, user_id: str) -> UserProfile:
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
        return self.profiles[user_id]

    def save_profile(self, profile: UserProfile):
        profile.last_seen = datetime.now().isoformat()
        self.profiles[profile.user_id] = profile
        self._save()


# =============================================================================
# MEMORY EXTRACTORS  (two interchangeable strategies)
# =============================================================================


class RuleBasedMemoryExtractor:
    """Deterministic keyword extraction. Fast, free, and unit-testable — no API
    call. Great for a first pass and for facts you can pattern-match reliably."""

    def extract(self, message: str) -> List[Memory]:
        memories = []
        msg = message.lower()

        if "my name is" in msg:
            memories.append(
                Memory(content=message, category="identity", importance=0.95)
            )
        if "prefer" in msg or "i like" in msg:
            memories.append(
                Memory(content=message, category="preference", importance=0.8)
            )
        if "$" in message or "budget" in msg:
            memories.append(
                Memory(content=message, category="financial", importance=0.85)
            )
        if any(w in msg for w in ["i work", "my job", "profession", "i am a", "i'm a"]):
            memories.append(
                Memory(content=message, category="profession", importance=0.75)
            )
        if any(w in msg for w in ["need", "require", "must have"]):
            memories.append(
                Memory(content=message, category="requirement", importance=0.8)
            )

        return memories


class LLMMemoryExtractor:
    """LLM-powered extraction. Claude reads the message and decides what's worth
    remembering — catches facts no keyword rule anticipated, at the cost of one
    API call per turn. Same `.extract()` interface as the rule-based version, so
    it's a drop-in swap."""

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.client = anthropic.Anthropic()
        self.model = model

    def extract(self, message: str) -> List[Memory]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=(
                "Extract durable facts worth remembering about the user from their "
                "message. Return ONLY a JSON array, no prose. Each item: "
                '{"content": "...", '
                '"category": "identity|preference|financial|profession|requirement|fact", '
                '"importance": 0.0-1.0}. '
                "If there's nothing worth remembering, return []."
            ),
            messages=[{"role": "user", "content": message}],
        )

        text = response.content[0].text.strip()
        try:
            facts = json.loads(text)
            return [
                Memory(
                    content=f["content"],
                    category=f.get("category", "fact"),
                    importance=float(f.get("importance", 0.5)),
                )
                for f in facts
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Model didn't return clean JSON — skip extraction this turn
            return []


# =============================================================================
# MEMORY RETRIEVER  (query-relevant + always-include)
# =============================================================================


class MemoryRetriever:
    """Pick which memories to inject into the current turn's context."""

    def __init__(self, profile: UserProfile):
        self.profile = profile

    def retrieve(self, query: str, max_memories: int = 8) -> List[Memory]:
        if not self.profile.memories:
            return []

        # Always include high-importance memories (name, budget, hard requirements)
        always = [m for m in self.profile.memories if m.importance >= 0.85]

        # Score the rest by simple word overlap with the query (a stand-in for
        # semantic search — see the 'Under the hood vs. managed' box in the lesson)
        query_words = set(query.lower().split())
        scored = []
        for mem in self.profile.memories:
            if mem in always:
                continue
            mem_words = set(mem.content.lower().split())
            overlap = len(query_words & mem_words)
            score = overlap / max(len(query_words), 1)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        remaining = max(0, max_memories - len(always))
        relevant = [m for _, m in scored[:remaining]]

        return always + relevant

    def format_context(self, memories: List[Memory]) -> str:
        if not memories:
            return ""

        by_cat: Dict[str, List[str]] = {}
        for m in memories:
            by_cat.setdefault(m.category, []).append(m.content)

        lines = ["## What I Remember About You"]
        for cat, items in by_cat.items():
            lines.append(f"\n### {cat.title()}")
            for item in items:
                lines.append(f"- {item}")

        return "\n".join(lines)


# =============================================================================
# COMPLETE CHATBOT
# =============================================================================


class MemoryEnabledChatbot:
    """Complete chatbot with memory. Pass any extractor with an `.extract()`
    method — defaults to the deterministic rule-based one."""

    def __init__(self, user_id: str, extractor=None, model: str = "claude-sonnet-5"):
        self.user_id = user_id
        self.model = model
        self.client = anthropic.Anthropic()

        # Initialize components
        self.store = MemoryStore()
        self.profile = self.store.get_profile(user_id)
        self.conversation = ConversationManager(max_turns=10)
        self.extractor = extractor or RuleBasedMemoryExtractor()
        self.retriever = MemoryRetriever(self.profile)

        # Update session count
        self.profile.session_count += 1

    def chat(self, user_message: str) -> str:
        # Add to conversation
        self.conversation.add_message("user", user_message)

        # Extract memories from this message
        for mem in self.extractor.extract(user_message):
            self.profile.add_memory(mem)

        # Capture the user's name if they introduced themselves
        self._maybe_capture_name(user_message)

        # Retrieve relevant memories and format them for context
        relevant = self.retriever.retrieve(user_message)
        memory_context = self.retriever.format_context(relevant)

        # Build the system prompt with memory context
        system = self._build_system_prompt(memory_context)

        # Get the response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=system,
            messages=self.conversation.get_context(),
        )

        assistant_msg = next(
            block.text for block in response.content if block.type == "text"
        )
        self.conversation.add_message("assistant", assistant_msg)
        return assistant_msg

    def _maybe_capture_name(self, user_message: str):
        lower = user_message.lower()
        marker = "my name is"
        if marker in lower:
            after = user_message[lower.index(marker) + len(marker) :].strip()
            if after:
                name = after.split()[0].strip(".,!?")
                if name:
                    self.profile.name = name.title()

    def _build_system_prompt(self, memory_context: str) -> str:
        greeting = (
            f"You are chatting with {self.profile.name}. " if self.profile.name else ""
        )
        session_info = (
            f"This is session #{self.profile.session_count} with this user. "
            if self.profile.session_count > 1
            else ""
        )

        return f"""You are a helpful assistant with memory of past conversations.
{greeting}{session_info}

{memory_context}

Use your memories to personalize responses, but don't be creepy about it.
Reference past information naturally when it's relevant."""

    def end_session(self) -> int:
        """Persist the profile at the end of a session."""
        self.store.save_profile(self.profile)
        return len(self.profile.memories)


# =============================================================================
# DEMO
# =============================================================================


def demo():
    print("=" * 60)
    print("MEMORY-ENABLED CHATBOT DEMO")
    print("=" * 60)

    # Clean up from previous runs so the demo is reproducible
    storage_path = "/tmp/chatbot_memory.json"
    if os.path.exists(storage_path):
        os.remove(storage_path)

    # SESSION 1 -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("SESSION 1")
    print("=" * 50)

    # Default = deterministic rule-based extractor (no extra API calls).
    # To use LLM-powered extraction instead:
    #     bot = MemoryEnabledChatbot("demo_user", extractor=LLMMemoryExtractor())
    bot = MemoryEnabledChatbot(user_id="demo_user")

    messages1 = [
        "Hi! My name is Sarah and I'm a software developer.",
        "I'm looking for a new laptop. My budget is around $1500.",
        "I prefer MacBooks but I'm open to other options.",
    ]

    for msg in messages1:
        print(f"\nUser: {msg}")
        response = bot.chat(msg)
        print(f"Bot: {response}")

    memories_saved = bot.end_session()
    print(f"\n[Session ended. {memories_saved} memories saved.]")

    # SESSION 2 (new session, memories persist) -----------------------------
    print("\n" + "=" * 50)
    print("SESSION 2 (NEW SESSION, SAME USER)")
    print("=" * 50)

    bot2 = MemoryEnabledChatbot(user_id="demo_user")
    print(
        f"\n[Loaded profile: {bot2.profile.name}, {len(bot2.profile.memories)} memories]"
    )

    messages2 = [
        "Hi again!",
        "Any laptop recommendations based on our last conversation?",
    ]

    for msg in messages2:
        print(f"\nUser: {msg}")
        response = bot2.chat(msg)
        print(f"Bot: {response}")

    bot2.end_session()

    # Show stored memories
    print("\n" + "=" * 50)
    print("STORED MEMORIES")
    print("=" * 50)
    for mem in bot2.profile.memories:
        print(f"  [{mem.category}] {mem.content}")

    # Cleanup
    if os.path.exists(storage_path):
        os.remove(storage_path)


if __name__ == "__main__":
    demo()
