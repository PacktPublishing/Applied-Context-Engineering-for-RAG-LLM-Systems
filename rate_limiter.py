"""
RATE LIMITING
Token bucket and sliding window rate limiters

Run: python rate_limiter.py
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
from typing import Dict

# =============================================================================
# TOKEN BUCKET
# =============================================================================


@dataclass
class TokenBucket:
    """Token bucket rate limiter - allows bursts up to capacity."""

    capacity: int
    refill_rate: float  # Tokens per second

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = datetime.now()
        self.lock = Lock()

    def _refill(self):
        """Add tokens based on time elapsed."""
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: int = 1) -> float:
        """How long to wait for tokens to be available."""
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                return 0
            needed = tokens - self.tokens
            return needed / self.refill_rate


# =============================================================================
# SLIDING WINDOW
# =============================================================================


class SlidingWindowLimiter:
    """Sliding window rate limiter - accurate but more memory."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests: deque = deque()
        self.lock = Lock()

    def _cleanup(self):
        """Remove requests outside the window."""
        cutoff = datetime.now() - self.window
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def allow(self) -> bool:
        """Check if request is allowed, recording it if so (atomic check+commit)."""
        with self.lock:
            self._cleanup()
            if len(self.requests) < self.max_requests:
                self.requests.append(datetime.now())
                return True
            return False

    def has_capacity(self) -> bool:
        """Check if a request WOULD be allowed, without recording it.

        Use with record() for check-then-commit across several limiters, so a
        request rejected by a *later* tier doesn't consume this tier's slot.
        """
        with self.lock:
            self._cleanup()
            return len(self.requests) < self.max_requests

    def record(self):
        """Record a request. Call only after every tier has passed has_capacity()."""
        with self.lock:
            self.requests.append(datetime.now())

    def remaining(self) -> int:
        """How many requests remaining in window."""
        with self.lock:
            self._cleanup()
            return max(0, self.max_requests - len(self.requests))

    def reset_time(self) -> float:
        """Seconds until oldest request expires."""
        with self.lock:
            self._cleanup()
            if not self.requests:
                return 0
            oldest = self.requests[0]
            reset_at = oldest + self.window
            return max(0, (reset_at - datetime.now()).total_seconds())


# =============================================================================
# TOKEN RATE LIMITER
# =============================================================================


class TokenRateLimiter:
    """Rate limit by token usage, not just requests."""

    def __init__(self, tokens_per_minute: int):
        self.tokens_per_minute = tokens_per_minute
        self.window: deque = deque()  # (timestamp, tokens)
        self.lock = Lock()

    def _cleanup(self):
        """Remove entries older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()

    def check(self, estimated_tokens: int) -> dict:
        """Check if token usage is allowed."""
        with self.lock:
            self._cleanup()
            current_usage = sum(t[1] for t in self.window)

            if current_usage + estimated_tokens > self.tokens_per_minute:
                return {
                    "allowed": False,
                    "current_usage": current_usage,
                    "limit": self.tokens_per_minute,
                    "requested": estimated_tokens,
                }

            return {
                "allowed": True,
                "current_usage": current_usage,
                "remaining": self.tokens_per_minute - current_usage,
            }

    def record(self, tokens: int):
        """Record token usage."""
        with self.lock:
            self.window.append((datetime.now(), tokens))

    def remaining(self) -> int:
        """Get remaining tokens."""
        with self.lock:
            self._cleanup()
            current_usage = sum(t[1] for t in self.window)
            return max(0, self.tokens_per_minute - current_usage)


# =============================================================================
# MULTI-TIER RATE LIMITER
# =============================================================================


class MultiTierRateLimiter:
    """Rate limit at multiple levels: global, per-user, per-operation."""

    def __init__(
        self,
        global_rpm: int = 1000,
        user_rpm: int = 60,
        tokens_per_minute: int = 100000,
    ):
        self.global_limiter = SlidingWindowLimiter(global_rpm, 60)
        self.token_limiter = TokenRateLimiter(tokens_per_minute)
        self.user_limiters: Dict[str, SlidingWindowLimiter] = {}
        self.user_rpm = user_rpm

    def check(self, user_id: str, estimated_tokens: int = 0) -> dict:
        """Check all rate limits, broadest first (check-then-commit).

        Every tier is checked *without* recording. Only if all tiers pass do we
        commit the request against the global and per-user windows — so a request
        rejected by a later tier never eats an earlier tier's slot.
        """
        # Lazily create this user's limiter
        if user_id not in self.user_limiters:
            self.user_limiters[user_id] = SlidingWindowLimiter(self.user_rpm, 60)
        user_limiter = self.user_limiters[user_id]

        # Check global (no commit yet)
        if not self.global_limiter.has_capacity():
            return {
                "allowed": False,
                "reason": "global_limit",
                "retry_after": self.global_limiter.reset_time(),
            }

        # Check user (no commit yet)
        if not user_limiter.has_capacity():
            return {
                "allowed": False,
                "reason": "user_limit",
                "retry_after": user_limiter.reset_time(),
            }

        # Check tokens (check() never records)
        if estimated_tokens > 0:
            token_check = self.token_limiter.check(estimated_tokens)
            if not token_check["allowed"]:
                return {
                    "allowed": False,
                    "reason": "token_limit",
                    "details": token_check,
                }

        # All tiers passed — commit the request slots now
        self.global_limiter.record()
        user_limiter.record()
        return {"allowed": True}

    def record_tokens(self, tokens: int):
        """Record token usage after successful request."""
        self.token_limiter.record(tokens)

    def get_status(self) -> dict:
        """Get current rate limit status."""
        return {
            "global_remaining": self.global_limiter.remaining(),
            "token_remaining": self.token_limiter.remaining(),
        }


# =============================================================================
# RATE LIMIT MIDDLEWARE
# =============================================================================


class RateLimitMiddleware:
    """Decorator for rate-limited functions."""

    def __init__(self, limiter: MultiTierRateLimiter):
        self.limiter = limiter

    def __call__(self, func):
        def wrapper(user_id: str, *args, **kwargs):
            check = self.limiter.check(user_id)
            if not check["allowed"]:
                raise RateLimitError(
                    reason=check["reason"], retry_after=check.get("retry_after", 60)
                )
            return func(user_id, *args, **kwargs)

        return wrapper


class RateLimitError(Exception):
    def __init__(self, reason: str, retry_after: float):
        self.reason = reason
        self.retry_after = retry_after
        super().__init__(f"Rate limited: {reason}")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("RATE LIMITING")
    print("=" * 60)

    # Demo 1: Token bucket
    print("\n" + "-" * 50)
    print("TOKEN BUCKET (10 tokens, 2/sec refill)")
    print("-" * 50)

    bucket = TokenBucket(capacity=10, refill_rate=2.0)

    # Burst of 10 requests
    for i in range(12):
        acquired = bucket.acquire(1)
        status = "OK" if acquired else f"WAIT {bucket.wait_time():.1f}s"
        print(f"Request {i+1}: {status}")

    # Demo 2: Sliding window
    print("\n" + "-" * 50)
    print("SLIDING WINDOW (5 requests per 10 seconds)")
    print("-" * 50)

    slider = SlidingWindowLimiter(max_requests=5, window_seconds=10)

    for i in range(7):
        allowed = slider.allow()
        status = "OK" if allowed else f"BLOCKED (retry in {slider.reset_time():.1f}s)"
        print(f"Request {i+1}: {status} (remaining: {slider.remaining()})")

    # Demo 3: Multi-tier
    print("\n" + "-" * 50)
    print("MULTI-TIER RATE LIMITER")
    print("-" * 50)

    multi = MultiTierRateLimiter(global_rpm=100, user_rpm=5, tokens_per_minute=50000)

    users = ["user_a", "user_a", "user_a", "user_a", "user_a", "user_a", "user_b"]
    for user in users:
        check = multi.check(user, estimated_tokens=1000)
        if check["allowed"]:
            multi.record_tokens(1000)
            print(f"{user}: ALLOWED")
        else:
            print(f"{user}: BLOCKED - {check['reason']}")

    print(f"\nStatus: {multi.get_status()}")

    # Summary
    print("\n" + "-" * 50)
    print("RATE LIMITING ALGORITHMS")
    print("-" * 50)
    print("""
| Algorithm       | Best For                | Trade-offs           |
|-----------------|-------------------------|----------------------|
| Token Bucket    | Bursts + steady rate    | Simple, allows burst |
| Sliding Window  | Accurate rate limiting  | More memory          |
| Multi-tier      | Complex applications    | Multiple limits      |
""")
