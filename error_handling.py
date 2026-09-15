"""
ERROR HANDLING
Robust error handling for context pipelines

Run: python error_handling.py
"""

from dataclasses import dataclass
from typing import Callable, Any, Optional, List
from enum import Enum
from datetime import datetime

# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================


class ErrorSeverity(Enum):
    TRANSIENT = "transient"  # Retry
    PERMANENT = "permanent"  # Stop
    PARTIAL = "partial"  # Degrade


@dataclass
class ContextError:
    stage: str
    error_type: str
    severity: ErrorSeverity
    message: str
    recoverable: bool
    fallback_available: bool
    timestamp: str = None

    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()


# =============================================================================
# ERROR HANDLER
# =============================================================================


class ContextErrorHandler:
    """Centralized error handling for context pipelines."""

    def __init__(self):
        self.error_log: List[ContextError] = []
        self.fallbacks: dict = {}

    def register_fallback(self, stage: str, fallback_fn: Callable):
        """Register a fallback for a pipeline stage."""
        self.fallbacks[stage] = fallback_fn

    def handle(self, stage: str, error: Exception) -> ContextError:
        """Handle an error and determine response."""
        error_type = type(error).__name__
        severity = self._classify_severity(error)

        context_error = ContextError(
            stage=stage,
            error_type=error_type,
            severity=severity,
            message=str(error)[:200],
            recoverable=severity != ErrorSeverity.PERMANENT,
            fallback_available=stage in self.fallbacks,
        )

        self.error_log.append(context_error)
        return context_error

    def _classify_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity."""
        error_msg = str(error).lower()

        # Transient - should retry
        if any(
            t in error_msg
            for t in ["rate", "timeout", "overload", "connection", "503", "429"]
        ):
            return ErrorSeverity.TRANSIENT

        # Permanent - don't retry
        if any(
            t in error_msg
            for t in ["auth", "invalid", "not found", "forbidden", "401", "400"]
        ):
            return ErrorSeverity.PERMANENT

        return ErrorSeverity.PARTIAL

    def execute_with_fallback(
        self, stage: str, primary_fn: Callable, *args, **kwargs
    ) -> tuple[Any, Optional[ContextError]]:
        """Execute with automatic fallback on failure."""
        try:
            result = primary_fn(*args, **kwargs)
            return result, None
        except Exception as e:
            error = self.handle(stage, e)

            if error.fallback_available:
                try:
                    fallback_result = self.fallbacks[stage](*args, **kwargs)
                    return fallback_result, error
                except Exception as fallback_error:
                    return None, self.handle(f"{stage}_fallback", fallback_error)

            return None, error

    def get_recent_errors(self, n: int = 10) -> List[ContextError]:
        """Get recent errors."""
        return self.error_log[-n:]


# =============================================================================
# STAGE-SPECIFIC HANDLERS
# =============================================================================


class RetrievalErrorHandler:
    """Handle retrieval-specific errors."""

    def __init__(self, handler: ContextErrorHandler):
        self.handler = handler
        handler.register_fallback("retrieval", self._fallback_retrieval)

    def retrieve_with_handling(
        self, query: str, retrieve_fn: Callable, min_results: int = 1
    ) -> list:
        """Retrieve with error handling."""
        result, error = self.handler.execute_with_fallback(
            "retrieval", retrieve_fn, query
        )

        if result is None:
            return self._fallback_retrieval(query)

        if len(result) < min_results:
            broader = self._broaden_query(query)
            additional, _ = self.handler.execute_with_fallback(
                "retrieval_broad", retrieve_fn, broader
            )
            if additional:
                result.extend(additional)

        return result

    def _fallback_retrieval(self, query: str) -> list:
        """Fallback: Return generic context."""
        return [
            {
                "content": "I don't have specific information about that topic.",
                "source": "fallback",
                "score": 0.0,
            }
        ]

    def _broaden_query(self, query: str) -> str:
        """Broaden query for more results."""
        words = query.split()
        return " ".join(words[:3])


class LLMErrorHandler:
    """Handle LLM-specific errors."""

    def __init__(self, handler: ContextErrorHandler):
        self.handler = handler
        handler.register_fallback("llm_call", self._fallback_response)

    def call_with_handling(self, messages: list, call_fn: Callable) -> str:
        """Call LLM with error handling."""
        result, error = self.handler.execute_with_fallback(
            "llm_call", call_fn, messages
        )

        if error:
            if error.severity == ErrorSeverity.TRANSIENT:
                raise TransientError(error.message)

        return result or self._fallback_response(messages)

    def _fallback_response(self, messages: list) -> str:
        """Fallback response when LLM fails."""
        return "I'm having trouble processing that request. Please try again."


class TransientError(Exception):
    """Error that should be retried."""

    pass


# =============================================================================
# USER-FACING ERROR MESSAGES
# =============================================================================


class UserFacingErrorHandler:
    """Convert internal errors to user-friendly messages."""

    ERROR_MESSAGES = {
        "rate_limit": "We're experiencing high demand. Please try again in a moment.",
        "context_length": "Your request is too complex. Try breaking it into smaller parts.",
        "retrieval_empty": "I couldn't find relevant information. Try rephrasing your question.",
        "timeout": "The request took too long. Please try a simpler question.",
        "default": "Something went wrong. Please try again.",
    }

    def get_user_message(self, error: ContextError) -> str:
        """Get user-friendly error message."""
        for key, message in self.ERROR_MESSAGES.items():
            if key == "default":
                continue
            # Keys use underscores ("rate_limit"); error text uses spaces
            # ("rate limit exceeded"), so match on both forms.
            needle = key.replace("_", " ")  # "Rate limit exceeded - 429"
            if needle in error.error_type.lower() or needle in error.message.lower():
                return message
        return self.ERROR_MESSAGES["default"]

    def should_show_retry(self, error: ContextError) -> bool:
        """Should we show a retry button?"""
        return error.severity == ErrorSeverity.TRANSIENT

    def get_suggestions(self, error: ContextError) -> list:
        """Get suggestions for the user."""
        if "context_length" in error.message.lower():
            return [
                "Ask a more specific question",
                "Break your request into parts",
                "Remove unnecessary details",
            ]
        elif "retrieval" in error.stage:
            return ["Try different keywords", "Be more specific", "Check for typos"]
        return []


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ERROR HANDLING")
    print("=" * 60)

    handler = ContextErrorHandler()
    retrieval_handler = RetrievalErrorHandler(handler)
    user_handler = UserFacingErrorHandler()

    # Demo 1: Handling different error types
    print("\n" + "-" * 50)
    print("ERROR CLASSIFICATION")
    print("-" * 50)

    errors = [
        Exception("Rate limit exceeded - 429"),
        Exception("Invalid API key - 401"),
        Exception("Connection timeout"),
        Exception("No results found"),
    ]

    for err in errors:
        ctx_error = handler.handle("test", err)
        print(f"\n{err}")
        print(f"  Type: {ctx_error.error_type}")
        print(f"  Severity: {ctx_error.severity.value}")
        print(f"  Recoverable: {ctx_error.recoverable}")

    # Demo 2: Fallback execution
    print("\n" + "-" * 50)
    print("FALLBACK EXECUTION")
    print("-" * 50)

    def failing_retrieval(query: str) -> list:
        raise Exception("Database connection failed")

    results = retrieval_handler.retrieve_with_handling("test query", failing_retrieval)
    print(f"Results (from fallback): {results}")

    # Demo 3: User-facing messages
    print("\n" + "-" * 50)
    print("USER-FACING MESSAGES")
    print("-" * 50)

    for err in handler.get_recent_errors():
        user_msg = user_handler.get_user_message(err)
        suggestions = user_handler.get_suggestions(err)
        print(f"\nInternal: {err.message}")
        print(f"User sees: {user_msg}")
        if suggestions:
            print(f"Suggestions: {suggestions}")

    # Summary
    print("\n" + "-" * 50)
    print("ERROR HANDLING SUMMARY")
    print("-" * 50)
    print(f"Total errors logged: {len(handler.error_log)}")
    print(
        f"Transient: {len([e for e in handler.error_log if e.severity == ErrorSeverity.TRANSIENT])}"
    )
    print(
        f"Permanent: {len([e for e in handler.error_log if e.severity == ErrorSeverity.PERMANENT])}"
    )
    print(
        f"Partial: {len([e for e in handler.error_log if e.severity == ErrorSeverity.PARTIAL])}"
    )
