"""
RETRY PATTERNS
Exponential backoff and circuit breakers

Run: python retry_patterns.py
"""

import time
import random
from dataclasses import dataclass
from typing import Callable, TypeVar, Optional
from datetime import datetime, timedelta
from enum import Enum

T = TypeVar("T")


# =============================================================================
# EXPONENTIAL BACKOFF
# =============================================================================


class ExponentialBackoff:
    """Retry with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def execute(
        self, func: Callable[[], T], retryable_exceptions: tuple = (Exception,)
    ) -> T:
        """Execute function with retries."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except retryable_exceptions as e:
                last_exception = e

                if attempt == self.max_retries:
                    break

                delay = self._calculate_delay(attempt)
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (1 + random.random() * 0.5)

        return delay


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker prevents cascading failures."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    def __post_init__(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def _should_attempt_recovery(self) -> bool:
        """Check if recovery timeout has passed."""
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0

    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            return

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def execute(self, func: Callable[[], T]) -> T:
        """Execute with circuit breaker protection."""
        if not self.can_execute():
            raise CircuitOpenError("Circuit breaker is open")

        try:
            result = func()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# =============================================================================
# ROBUST RETRY
# =============================================================================


class RobustRetry:
    """Combined retry with backoff and circuit breaker."""

    RETRYABLE_PATTERNS = ["rate", "timeout", "overload", "connection", "503", "429"]

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, failure_threshold: int = 5
    ):
        self.backoff = ExponentialBackoff(
            max_retries=max_retries, base_delay=base_delay
        )
        self.circuit = CircuitBreaker(
            failure_threshold=failure_threshold, recovery_timeout=30.0
        )

    def execute(self, func: Callable[[], T]) -> T:
        """Execute with full retry protection."""
        if not self.circuit.can_execute():
            raise CircuitOpenError("Service unavailable - circuit open")

        last_exception = None

        for attempt in range(self.backoff.max_retries + 1):
            try:
                result = func()
                self.circuit.record_success()
                return result

            except Exception as e:
                last_exception = e

                if not self._is_retryable(e):
                    self.circuit.record_failure()
                    raise

                if attempt == self.backoff.max_retries:
                    self.circuit.record_failure()
                    break

                delay = self.backoff._calculate_delay(attempt)
                time.sleep(delay)

        raise last_exception

    def _is_retryable(self, error: Exception) -> bool:
        """Determine if error is retryable."""
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.RETRYABLE_PATTERNS)

    def get_status(self) -> dict:
        """Get retry system status."""
        return {
            "circuit_state": self.circuit.state.value,
            "failure_count": self.circuit.failure_count,
        }


# =============================================================================
# RETRY DECORATOR
# =============================================================================


def with_retry(
    max_retries: int = 3, base_delay: float = 1.0, retryable: tuple = (Exception,)
):
    """Decorator for adding retry logic."""

    def decorator(func):
        backoff = ExponentialBackoff(max_retries=max_retries, base_delay=base_delay)

        def wrapper(*args, **kwargs):
            return backoff.execute(
                lambda: func(*args, **kwargs), retryable_exceptions=retryable
            )

        return wrapper

    return decorator


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RETRY PATTERNS")
    print("=" * 60)

    # Demo 1: Exponential backoff
    print("\n" + "-" * 50)
    print("EXPONENTIAL BACKOFF")
    print("-" * 50)

    backoff = ExponentialBackoff(max_retries=3, base_delay=0.5)

    call_count = 0

    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "Success!"

    try:
        result = backoff.execute(flaky_function)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    # Demo 2: Circuit breaker
    print("\n" + "-" * 50)
    print("CIRCUIT BREAKER")
    print("-" * 50)

    circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

    def failing_service():
        raise Exception("Service down")

    for i in range(7):
        try:
            circuit.execute(failing_service)
        except CircuitOpenError:
            print(f"Call {i+1}: REJECTED (circuit open)")
        except Exception as e:
            print(f"Call {i+1}: FAILED ({circuit.state.value})")

    print(f"\nWaiting for recovery timeout...")
    time.sleep(2.5)

    # Circuit should be half-open now
    print(f"Circuit state: {circuit.state.value}")

    # Demo 3: Combined retry
    print("\n" + "-" * 50)
    print("ROBUST RETRY (Backoff + Circuit Breaker)")
    print("-" * 50)

    retry = RobustRetry(max_retries=2, base_delay=0.5, failure_threshold=3)

    def simulate_api():
        if random.random() < 0.6:
            raise Exception("Rate limit exceeded - 429")
        return "API response"

    for i in range(5):
        try:
            result = retry.execute(simulate_api)
            print(f"Call {i+1}: SUCCESS - {result}")
        except CircuitOpenError:
            print(f"Call {i+1}: CIRCUIT OPEN")
        except Exception as e:
            print(f"Call {i+1}: FAILED - {e}")

    print(f"\nStatus: {retry.get_status()}")

    # Summary
    print("\n" + "-" * 50)
    print("RETRY PATTERNS SUMMARY")
    print("-" * 50)
    print("""
| Pattern            | Use Case                     |
|--------------------|------------------------------|
| Exponential Backoff| Transient failures           |
| Circuit Breaker    | Prevent cascade failures     |
| Combined           | Production systems           |
| Decorator          | Simple retry needs           |

Key principles:
1. Always add jitter to prevent thundering herd
2. Circuit breaker stops retrying when service is down
3. Only retry transient errors (rate limits, timeouts)
4. Set reasonable max retries (3-5 typically)
""")
