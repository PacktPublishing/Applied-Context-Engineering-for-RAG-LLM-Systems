"""
MEMORY INJECTION PATTERNS
How to inject retrieved memories into context

Run: python injection_patterns.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class Memory:
    content: str
    category: str
    importance: float = 0.5
    recency: float = 0.5  # 0 = old, 1 = recent


@dataclass
class UserProfile:
    name: str
    preferences: Dict[str, str] = field(default_factory=dict)
    facts: List[str] = field(default_factory=list)
    total_sessions: int = 0
    created_at: datetime = None
    patterns: Dict[str, str] = field(default_factory=dict)


def format_memories(memories: List[Memory]) -> str:
    """Group memories by category into clean markdown."""
    by_category: Dict[str, List[str]] = {}
    for mem in memories:
        by_category.setdefault(mem.category, []).append(mem.content)

    lines = []
    for cat, items in by_category.items():
        lines.append(f"### {cat.replace('_', ' ').title()}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).rstrip()


# =============================================================================
# PATTERN 1: SYSTEM CONTEXT INJECTION
# =============================================================================


def inject_as_system_context(system_prompt: str, memories: List[Memory]) -> str:
    """
    Inject memories into the system prompt.
    Default choice - works for most cases.
    """
    memory_section = format_memories(memories)

    return f"""{system_prompt}

## User Context (Retrieved Memories)
{memory_section}

Use this context to personalize your responses. Reference user information naturally when relevant."""


# =============================================================================
# PATTERN 2: INLINE INJECTION
# =============================================================================


def inject_inline(
    conversation: List[dict], relevant_memories: List[Memory]
) -> List[dict]:
    """
    Inject memories inline, right before the current user turn.
    Good when memories are specifically relevant to the current turn.
    """
    if not conversation or conversation[-1]["role"] != "user":
        return conversation

    # Format memories as a context reminder
    memory_text = "\n".join(f"• {m.content}" for m in relevant_memories)

    # Enhance the last user message with context
    last_message = conversation[-1]
    enhanced = {
        "role": "user",
        "content": f"[Context reminder:\n{memory_text}]\n\n{last_message['content']}",
    }

    return conversation[:-1] + [enhanced]


# =============================================================================
# PATTERN 3: PERSONA INJECTION
# =============================================================================


def inject_as_persona(base_persona: str, profile: UserProfile) -> str:
    """
    Frame memories as part of the AI's knowledge of the user.
    Good for personal assistants and long-running relationships.
    """
    facts_text = "\n".join(f"  - {f}" for f in profile.facts[-5:])
    pref_text = "\n".join(f"  - {k}: {v}" for k, v in profile.preferences.items())

    return f"""{base_persona}

You are assisting {profile.name}, who you know well from {profile.total_sessions} previous conversations.

What you know about {profile.name}:
{facts_text}

Their preferences:
{pref_text}

Adapt your responses to their known preferences and context.
Reference past conversations naturally when relevant."""


# =============================================================================
# PATTERN 4: CONTEXTUAL REMINDER
# =============================================================================


def inject_contextual_reminder(
    conversation: List[dict], memories: List[Memory], trigger_turn: int
) -> List[dict]:
    """
    Insert a reminder at a specific point in the conversation.
    Good when the topic shifts and old memories become relevant.
    """
    if trigger_turn > len(conversation):
        return conversation

    memory_text = ", ".join(m.content for m in memories[:3])

    reminder = {
        "role": "assistant",
        "content": f"[I recall from our previous conversations: {memory_text}]",
    }

    return conversation[:trigger_turn] + [reminder] + conversation[trigger_turn:]


# =============================================================================
# PATTERN 5: STRUCTURED PROFILE INJECTION
# =============================================================================


def inject_structured_profile(profile: UserProfile) -> str:
    """
    Highly formatted, complete profile injection.
    Good for complex applications with rich user data.
    """
    created = (
        profile.created_at.strftime("%Y-%m-%d") if profile.created_at else "unknown"
    )
    prefs = "\n".join(f"- **{k}**: {v}" for k, v in profile.preferences.items())
    facts = "\n".join(f"- {f}" for f in profile.facts[-5:])
    comm_style = profile.patterns.get("communication_style", "unknown")
    peak = profile.patterns.get("peak_activity_hour", "unknown")

    return f"""## User Profile

| Field | Value |
|-------|-------|
| Name | {profile.name} |
| Sessions | {profile.total_sessions} |
| Member Since | {created} |

### Preferences
{prefs}

### Key Facts
{facts}

### Behavioral Patterns
- Communication style: {comm_style}
- Peak activity: {peak}
"""


# =============================================================================
# BEST PRACTICES
# =============================================================================


class MemoryInjectionBestPractices:
    """
    1. Position: Put memories BEFORE the task, not after
    2. Format: Use clear headers and structure
    3. Relevance: Only inject relevant memories
    4. Freshness: Note when memories are old
    5. Attribution: Make clear these are remembered facts
    """

    @staticmethod
    def format_with_freshness(memories: List[Memory]) -> str:
        """Include freshness indicators."""
        lines = []
        for mem in memories:
            freshness = "recent" if mem.recency > 0.8 else "older"
            lines.append(f"- [{freshness}] {mem.content}")
        return "\n".join(lines)

    @staticmethod
    def add_uncertainty_markers(memories: List[Memory]) -> str:
        """Mark low-confidence memories."""
        lines = []
        for mem in memories:
            if mem.importance < 0.7:
                lines.append(f"- (possibly) {mem.content}")
            else:
                lines.append(f"- {mem.content}")
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY INJECTION PATTERNS")
    print("=" * 60)

    # Sample data
    memories = [
        Memory("User's name is Alex", "identity", importance=0.95, recency=0.9),
        Memory("Prefers concise responses", "preference", importance=0.8, recency=0.7),
        Memory("Works in data science", "profession", importance=0.75, recency=0.5),
        Memory("Budget is $2000", "financial", importance=0.85, recency=0.95),
    ]

    profile = UserProfile(
        name="Alex",
        preferences={
            "communication": "concise",
            "tone": "professional",
            "format": "bullet points",
        },
        facts=[
            "Works as a data scientist",
            "Prefers MacBooks",
            "Travels frequently",
            "Uses Python daily",
        ],
        total_sessions=5,
        created_at=datetime(2025, 1, 15),
        patterns={"communication_style": "direct", "peak_activity_hour": "9am-11am"},
    )

    conversation = [
        {"role": "user", "content": "Hi, I need laptop advice."},
        {"role": "assistant", "content": "I'd be happy to help! What's your budget?"},
        {"role": "user", "content": "What would you recommend for machine learning?"},
    ]

    # Pattern 1: System Context
    print("\n" + "-" * 50)
    print("PATTERN 1: SYSTEM CONTEXT INJECTION")
    print("-" * 50)
    print(inject_as_system_context("You are a helpful assistant.", memories))

    # Pattern 2: Inline
    print("\n" + "-" * 50)
    print("PATTERN 2: INLINE INJECTION")
    print("-" * 50)
    result2 = inject_inline(conversation, memories[:2])
    print("Last message enhanced:")
    print(result2[-1]["content"])

    # Pattern 3: Persona
    print("\n" + "-" * 50)
    print("PATTERN 3: PERSONA INJECTION")
    print("-" * 50)
    print(inject_as_persona("You are a tech advisor.", profile))

    # Pattern 4: Contextual Reminder
    print("\n" + "-" * 50)
    print("PATTERN 4: CONTEXTUAL REMINDER")
    print("-" * 50)
    result4 = inject_contextual_reminder(conversation, memories[:2], trigger_turn=2)
    print("Conversation with reminder:")
    for msg in result4:
        print(f"  {msg['role']}: {msg['content'][:60]}")

    # Pattern 5: Structured Profile
    print("\n" + "-" * 50)
    print("PATTERN 5: STRUCTURED PROFILE")
    print("-" * 50)
    print(inject_structured_profile(profile))

    # Best Practices
    print("\n" + "-" * 50)
    print("BEST PRACTICES: FRESHNESS + UNCERTAINTY MARKERS")
    print("-" * 50)
    print("With freshness indicators:")
    print(MemoryInjectionBestPractices.format_with_freshness(memories))
    print("\nWith uncertainty markers:")
    print(MemoryInjectionBestPractices.add_uncertainty_markers(memories))

    # Summary
    print("\n" + "=" * 60)
    print("WHEN TO USE EACH PATTERN")
    print("=" * 60)
    print("""
| Pattern              | Best For                                |
|----------------------|-----------------------------------------|
| System Context       | Default choice, most applications       |
| Inline               | Query-specific relevant memories        |
| Persona              | Personal assistants, long relationships |
| Contextual Reminder  | Topic shifts, bringing back old context |
| Structured Profile   | Rich user data, complex applications    |
""")
