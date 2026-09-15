"""
LONG-TERM MEMORY
Persistent user profiles across sessions

Run: python long_term_memory.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json
import os


@dataclass
class Memory:
    content: str
    category: str
    importance: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserProfile:
    user_id: str
    name: Optional[str] = None
    memories: List[Memory] = field(default_factory=list)
    preferences: Dict[str, str] = field(default_factory=dict)
    session_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_memory(self, memory: Memory):
        """Add memory if not duplicate."""
        existing = [m.content for m in self.memories]
        if memory.content not in existing:
            self.memories.append(memory)

    def set_preference(self, key: str, value: str):
        self.preferences[key] = value

    def get_memories_by_category(self, category: str) -> List[Memory]:
        return [m for m in self.memories if m.category == category]

    def to_context(self) -> str:
        """Format profile for LLM context."""
        lines = []

        if self.name:
            lines.append(f"## User: {self.name}")
        else:
            lines.append("## User Profile")

        lines.append(f"Sessions: {self.session_count}")

        if self.preferences:
            lines.append("\n### Preferences")
            for key, value in self.preferences.items():
                lines.append(f"- {key}: {value}")

        if self.memories:
            lines.append("\n### Known Facts")
            # Sort by importance, show top 10
            sorted_mems = sorted(
                self.memories, key=lambda m: m.importance, reverse=True
            )
            for mem in sorted_mems[:10]:
                lines.append(f"- [{mem.category}] {mem.content}")

        return "\n".join(lines)


class LongTermMemoryStore:
    """Persist user profiles to JSON (use a database in production)."""

    def __init__(self, storage_path: str = "ltm_store.json"):
        self.storage_path = storage_path
        self.profiles: Dict[str, UserProfile] = {}
        self._load()

    def _load(self):
        """Load profiles from storage."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            for user_id, pdata in data.items():
                memories = [
                    Memory(
                        content=m["content"],
                        category=m["category"],
                        importance=m.get("importance", 0.5),
                        created_at=m.get("created_at", ""),
                    )
                    for m in pdata.get("memories", [])
                ]

                self.profiles[user_id] = UserProfile(
                    user_id=user_id,
                    name=pdata.get("name"),
                    memories=memories,
                    preferences=pdata.get("preferences", {}),
                    session_count=pdata.get("session_count", 0),
                    created_at=pdata.get("created_at", ""),
                    last_seen=pdata.get("last_seen", ""),
                )
        except Exception as e:
            print(f"Error loading LTM: {e}")

    def _save(self):
        """Save profiles to storage."""
        data = {}
        for user_id, profile in self.profiles.items():
            data[user_id] = {
                "name": profile.name,
                "memories": [
                    {
                        "content": m.content,
                        "category": m.category,
                        "importance": m.importance,
                        "created_at": m.created_at,
                    }
                    for m in profile.memories
                ],
                "preferences": profile.preferences,
                "session_count": profile.session_count,
                "created_at": profile.created_at,
                "last_seen": profile.last_seen,
            }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
        return self.profiles[user_id]

    def save_profile(self, profile: UserProfile):
        """Save profile to storage."""
        profile.last_seen = datetime.now().isoformat()
        self.profiles[profile.user_id] = profile
        self._save()


class MemoryExtractor:
    """Extract memorable facts from messages."""

    def extract(self, message: str) -> List[Memory]:
        """
        Extract facts worth remembering long-term.
        In production, use an LLM for this.
        """
        memories = []
        message_lower = message.lower()

        # Name extraction
        if "my name is" in message_lower:
            memories.append(
                Memory(content=message, category="identity", importance=0.95)
            )

        # Preference extraction
        if "prefer" in message_lower or "i like" in message_lower:
            memories.append(
                Memory(content=message, category="preference", importance=0.8)
            )

        # Role/profession
        if any(w in message_lower for w in ["i work", "i'm a", "my job", "i am a"]):
            memories.append(
                Memory(content=message, category="profession", importance=0.75)
            )

        # Budget/financial
        if "$" in message or "budget" in message_lower:
            memories.append(
                Memory(content=message, category="financial", importance=0.85)
            )

        # Goals
        if any(
            w in message_lower for w in ["trying to", "want to", "goal is", "hoping to"]
        ):
            memories.append(Memory(content=message, category="goal", importance=0.7))

        return memories


# =============================================================================
# DEMO
# =============================================================================


def demo_session_1(store: LongTermMemoryStore, extractor: MemoryExtractor):
    """Simulate first session with a user."""
    print("\n" + "=" * 50)
    print("SESSION 1")
    print("=" * 50)

    profile = store.get_profile("user_123")
    profile.session_count += 1

    messages = [
        "Hi! My name is Alex and I'm a data scientist.",
        "I'm looking for a good laptop for machine learning work.",
        "My budget is around $2500.",
        "I prefer lightweight laptops because I travel a lot.",
    ]

    print("\nConversation:")
    for msg in messages:
        print(f"  User: {msg}")

        # Extract and store memories
        new_memories = extractor.extract(msg)
        for mem in new_memories:
            profile.add_memory(mem)
            print(f"    → Stored: [{mem.category}] (importance: {mem.importance})")

        # Extract name specifically
        if "my name is" in msg.lower():
            name = msg.split("my name is")[-1].split()[0].strip(".,!")
            profile.name = name.title()

    # Save at end of session
    store.save_profile(profile)
    print(f"\nSession ended. Profile saved with {len(profile.memories)} memories.")


def demo_session_2(store: LongTermMemoryStore):
    """Simulate second session - memories should persist."""
    print("\n" + "=" * 50)
    print("SESSION 2 (MEMORIES SHOULD PERSIST)")
    print("=" * 50)

    profile = store.get_profile("user_123")
    profile.session_count += 1

    print(f"\nLoaded profile for: {profile.name or 'Unknown'}")
    print(f"Previous sessions: {profile.session_count - 1}")
    print(f"Stored memories: {len(profile.memories)}")

    print("\nContext for LLM:")
    print("-" * 40)
    print(profile.to_context())
    print("-" * 40)

    # Save updated session count
    store.save_profile(profile)


if __name__ == "__main__":
    print("=" * 60)
    print("LONG-TERM MEMORY DEMONSTRATION")
    print("=" * 60)

    # Use temp file for demo
    storage_path = "/tmp/ltm_demo.json"

    # Clean up from previous runs
    if os.path.exists(storage_path):
        os.remove(storage_path)

    store = LongTermMemoryStore(storage_path)
    extractor = MemoryExtractor()

    # Simulate two sessions
    demo_session_1(store, extractor)
    demo_session_2(store)

    print("\n" + "=" * 60)
    print("KEY POINTS")
    print("=" * 60)
    print("""
1. Session 1: User shares info, we extract and store memories
2. Session 2: We load persisted memories - user is recognized!
3. Memory categories: identity, preference, profession, financial, goal
4. Importance scores prioritize which memories to show
5. In production: use a database, not JSON files
""")

    # Cleanup
    if os.path.exists(storage_path):
        os.remove(storage_path)
