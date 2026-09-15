"""
CONTEXT CACHING
Multi-layer caching for context systems

Run: python context_cache.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import hashlib
import json

# =============================================================================
# RESPONSE CACHE
# =============================================================================


@dataclass
class CacheEntry:
    value: Any
    created: datetime
    expires: datetime
    hits: int = 0


class ResponseCache:
    """Cache LLM responses by context hash."""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 10000):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_entries = max_entries
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {"hits": 0, "misses": 0}

    def _make_key(self, messages: list, system: str = "") -> str:
        """Create cache key from request."""
        content = json.dumps({"system": system, "messages": messages}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, messages: list, system: str = "") -> Optional[str]:
        """Get cached response."""
        key = self._make_key(messages, system)

        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry.expires:
                entry.hits += 1
                self.stats["hits"] += 1
                return entry.value
            else:
                del self.cache[key]

        self.stats["misses"] += 1
        return None

    def set(self, messages: list, system: str, response: str):
        """Cache a response."""
        # Evict if at capacity
        if len(self.cache) >= self.max_entries:
            self._evict_oldest()

        key = self._make_key(messages, system)
        now = datetime.now()
        self.cache[key] = CacheEntry(
            value=response, created=now, expires=now + self.ttl
        )

    def _evict_oldest(self):
        """Evict oldest entry."""
        if not self.cache:
            return
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created)
        del self.cache[oldest_key]

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            "entries": len(self.cache),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
        }


# =============================================================================
# EMBEDDING CACHE
# =============================================================================


class EmbeddingCache:
    """Cache embeddings permanently (they never change)."""

    def __init__(self):
        self.cache: Dict[str, list] = {}
        self.stats = {"hits": 0, "computes": 0}

    def _text_hash(self, text: str) -> str:
        """Hash text for cache key."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def get(self, text: str) -> Optional[list]:
        """Get cached embedding."""
        key = self._text_hash(text)
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]
        return None

    def set(self, text: str, embedding: list):
        """Cache embedding."""
        key = self._text_hash(text)
        self.cache[key] = embedding
        self.stats["computes"] += 1

    def get_or_compute(self, text: str, compute_fn) -> list:
        """Get from cache or compute."""
        cached = self.get(text)
        if cached is not None:
            return cached

        embedding = compute_fn(text)
        self.set(text, embedding)
        return embedding

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["computes"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            "entries": len(self.cache),
            "hits": self.stats["hits"],
            "computes": self.stats["computes"],
            "hit_rate": f"{hit_rate:.1%}",
        }


# =============================================================================
# MULTI-LAYER CACHE
# =============================================================================


class MultiLayerCache:
    """L1 (memory) + L2 (persistent) caching."""

    def __init__(self, l1_max_size: int = 1000):
        self.l1_max_size = l1_max_size
        self.l1_cache: Dict[str, dict] = {}  # Fast, limited
        self.l2_cache: Dict[str, dict] = {}  # Slower, larger
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}

    def get(self, key: str) -> Optional[Any]:
        """Get from L1, then L2."""
        # Check L1
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if datetime.now() < entry["expires"]:
                entry["accessed"] = datetime.now()
                self.stats["l1_hits"] += 1
                return entry["value"]
            del self.l1_cache[key]

        # Check L2
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if datetime.now() < entry["expires"]:
                # Promote to L1
                self._set_l1(key, entry["value"], entry["expires"])
                self.stats["l2_hits"] += 1
                return entry["value"]
            del self.l2_cache[key]

        self.stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set in both layers."""
        expires = datetime.now() + timedelta(seconds=ttl)
        self._set_l1(key, value, expires)
        self._set_l2(key, value, expires)

    def _set_l1(self, key: str, value: Any, expires: datetime):
        """Set in L1 with LRU eviction."""
        if len(self.l1_cache) >= self.l1_max_size:
            oldest = min(
                self.l1_cache.keys(), key=lambda k: self.l1_cache[k]["accessed"]
            )
            del self.l1_cache[oldest]

        self.l1_cache[key] = {
            "value": value,
            "expires": expires,
            "accessed": datetime.now(),
        }

    def _set_l2(self, key: str, value: Any, expires: datetime):
        """Set in L2."""
        self.l2_cache[key] = {"value": value, "expires": expires}

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["misses"]
        return {
            "l1_entries": len(self.l1_cache),
            "l2_entries": len(self.l2_cache),
            "l1_hits": self.stats["l1_hits"],
            "l2_hits": self.stats["l2_hits"],
            "misses": self.stats["misses"],
            "hit_rate": (
                f"{(self.stats['l1_hits'] + self.stats['l2_hits']) / total:.1%}"
                if total > 0
                else "0%"
            ),
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CONTEXT CACHING")
    print("=" * 60)

    # Demo 1: Response cache
    print("\n" + "-" * 50)
    print("RESPONSE CACHE")
    print("-" * 50)

    response_cache = ResponseCache(ttl_seconds=60)

    # Simulate requests
    messages = [{"role": "user", "content": "What is AI?"}]
    system = "You are a helpful assistant."

    # First request - cache miss
    result = response_cache.get(messages, system)
    print(f"First request: {result}")  # None

    # Store response
    response_cache.set(messages, system, "AI is artificial intelligence...")

    # Second request - cache hit
    result = response_cache.get(messages, system)
    print(f"Second request: {result[:50]}...")  # Cached

    # Different messages - cache miss
    other_messages = [{"role": "user", "content": "What is ML?"}]
    result = response_cache.get(other_messages, system)
    print(f"Different query: {result}")  # None

    print(f"\nStats: {response_cache.get_stats()}")

    # Demo 2: Embedding cache
    print("\n" + "-" * 50)
    print("EMBEDDING CACHE")
    print("-" * 50)

    embedding_cache = EmbeddingCache()

    def mock_embed(text: str) -> list:
        """Simulate expensive embedding."""
        print(f"  Computing embedding for: {text[:30]}...")
        return [0.1, 0.2, 0.3]  # Fake embedding

    # First time - computes
    emb1 = embedding_cache.get_or_compute("Hello world", mock_embed)

    # Second time - cached
    emb2 = embedding_cache.get_or_compute("Hello world", mock_embed)

    # Same content, different case - cached
    emb3 = embedding_cache.get_or_compute("HELLO WORLD", mock_embed)

    print(f"\nStats: {embedding_cache.get_stats()}")

    # Demo 3: Multi-layer cache
    print("\n" + "-" * 50)
    print("MULTI-LAYER CACHE")
    print("-" * 50)

    ml_cache = MultiLayerCache(l1_max_size=5)

    # Store 10 items. L1 holds only the 5 most-recently-set keys (key_5..key_9);
    # L2 keeps all 10.
    for i in range(10):
        ml_cache.set(f"key_{i}", f"value_{i}", ttl=3600)

    # Read the 5 hottest keys first -> served fast from L1.
    # Then read the 5 colder keys -> L2 hits, each promoted back up to L1.
    read_order = list(range(9, 4, -1)) + list(range(5))
    for i in read_order:
        before = dict(ml_cache.stats)
        result = ml_cache.get(f"key_{i}")
        if ml_cache.stats["l1_hits"] > before["l1_hits"]:
            layer = "L1"
        elif ml_cache.stats["l2_hits"] > before["l2_hits"]:
            layer = "L2"
        else:
            layer = "miss"
        print(f"key_{i}: {layer}")

    print(f"\nStats: {ml_cache.get_stats()}")

    # Summary
    print("\n" + "-" * 50)
    print("CACHING BENEFITS")
    print("-" * 50)
    print("""
| Cache Type     | Use Case                    | TTL      |
|----------------|-----------------------------| ---------|
| Response       | Same query = same response  | 1 hour   |
| Embedding      | Text → vector (permanent)   | Forever  |
| Multi-layer    | Hot data + warm data        | Variable |
""")
