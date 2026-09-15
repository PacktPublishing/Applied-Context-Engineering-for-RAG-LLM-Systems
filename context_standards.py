"""
CONTEXT ENGINEERING STANDARDS
Team conventions, style guides, and best practices

Run: python context_standards.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from abc import ABC, abstractmethod

# =============================================================================
# PRINCIPLES
# =============================================================================


class ContextPrinciple(Enum):
    """Core principles for context engineering."""

    RELEVANT_OVER_COMPLETE = "relevant_over_complete"
    EXPLICIT_SOURCES = "explicit_sources"
    VERSION_EVERYTHING = "version_everything"
    SECURE_BY_DEFAULT = "secure_by_default"
    OBSERVABLE = "observable"
    TESTABLE = "testable"


@dataclass
class TeamPrinciples:
    """Team's context engineering principles."""

    principles: List[ContextPrinciple]
    descriptions: Dict[ContextPrinciple, str]

    @classmethod
    def default(cls) -> "TeamPrinciples":
        return cls(
            principles=list(ContextPrinciple),
            descriptions={
                ContextPrinciple.RELEVANT_OVER_COMPLETE: "Include only context that helps answer the query.",
                ContextPrinciple.EXPLICIT_SOURCES: "Always track where context came from.",
                ContextPrinciple.VERSION_EVERYTHING: "Context schemas and prompts are versioned.",
                ContextPrinciple.SECURE_BY_DEFAULT: "PII is redacted by default.",
                ContextPrinciple.OBSERVABLE: "Every operation emits metrics and logs.",
                ContextPrinciple.TESTABLE: "Context quality is tested automatically.",
            },
        )


# =============================================================================
# NAMING CONVENTIONS
# =============================================================================


@dataclass
class NamingConventions:
    """Naming standards for context systems."""

    CONTEXT_TYPES = {
        "system": "SystemContext",
        "user": "UserContext",
        "session": "SessionContext",
        "document": "DocumentContext",
        "retrieved": "RetrievedContext",
    }

    METRIC_PATTERNS = {
        "tokens": "context.tokens.{stage}",
        "latency": "context.latency.{operation}",
        "quality": "context.quality.{metric}",
        "cost": "context.cost.{dimension}",
    }

    REQUIRED_LOG_FIELDS = ["request_id", "timestamp", "context_version", "operation"]


class ContextPrefix:
    """Standard prefixes for naming."""

    SYSTEM = "sys_"
    USER = "usr_"
    DOC = "doc_"
    CACHE = "cache_"
    METRIC = "metric_"


# =============================================================================
# STANDARD INTERFACES
# =============================================================================


@dataclass
class ContextResult:
    """Standard result format for context retrieval."""

    content: str
    sources: List[str]
    metadata: Dict[str, Any]
    version: str
    timestamp: str
    quality_score: Optional[float] = None


@dataclass
class ValidationResult:
    """Standard result format for validation."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextProvider(ABC):
    """Standard interface all context providers must implement."""

    @abstractmethod
    def retrieve(self, query: str, config: Dict[str, Any]) -> ContextResult:
        """Retrieve context for a query."""
        pass

    @abstractmethod
    def validate(self, context: ContextResult) -> ValidationResult:
        """Validate retrieved context."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return provider metadata."""
        pass


# =============================================================================
# STANDARDS ENFORCEMENT
# =============================================================================


@dataclass
class RuleResult:
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class StandardsRule:
    """A single standards rule."""

    name: str
    description: str
    artifact_types: List[str]
    check_fn: Callable
    severity: str = "error"

    def applies_to(self, artifact_type: str) -> bool:
        return artifact_type in self.artifact_types

    def check(self, artifact: Any) -> RuleResult:
        return self.check_fn(artifact)


@dataclass
class CheckResult:
    passed: bool
    violations: List[str]
    warnings: List[str]


class StandardsChecker:
    """Enforce context engineering standards."""

    def __init__(self):
        self.rules: List[StandardsRule] = []

    def add_rule(self, rule: StandardsRule):
        self.rules.append(rule)

    def check(self, artifact: Any, artifact_type: str) -> CheckResult:
        violations = []
        warnings = []

        for rule in self.rules:
            if rule.applies_to(artifact_type):
                result = rule.check(artifact)
                violations.extend(result.violations)
                warnings.extend(result.warnings)

        return CheckResult(
            passed=len(violations) == 0, violations=violations, warnings=warnings
        )


# Standard rule implementations
def check_context_has_sources(context: ContextResult) -> RuleResult:
    result = RuleResult()
    if not context.sources:
        result.violations.append("Context must have at least one source")
    return result


def check_context_has_version(context: ContextResult) -> RuleResult:
    result = RuleResult()
    if not context.version:
        result.violations.append("Context must have a version")
    return result


def check_quality_score_present(context: ContextResult) -> RuleResult:
    result = RuleResult()
    if context.quality_score is None:
        result.warnings.append("Context should include quality score")
    return result


def create_standard_checker() -> StandardsChecker:
    """Create checker with standard rules."""
    checker = StandardsChecker()

    checker.add_rule(
        StandardsRule(
            name="sources_required",
            description="All context must have traceable sources",
            artifact_types=["context"],
            check_fn=check_context_has_sources,
        )
    )

    checker.add_rule(
        StandardsRule(
            name="version_required",
            description="All context must be versioned",
            artifact_types=["context"],
            check_fn=check_context_has_version,
        )
    )

    checker.add_rule(
        StandardsRule(
            name="quality_score_recommended",
            description="Context should include quality scores",
            artifact_types=["context"],
            check_fn=check_quality_score_present,
            severity="warning",
        )
    )

    return checker


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CONTEXT ENGINEERING STANDARDS")
    print("=" * 60)

    # Show principles
    print("\n" + "-" * 50)
    print("TEAM PRINCIPLES")
    print("-" * 50)

    principles = TeamPrinciples.default()
    for principle in principles.principles:
        print(f"  {principle.name}: {principles.descriptions[principle]}")

    # Show naming conventions
    print("\n" + "-" * 50)
    print("NAMING CONVENTIONS")
    print("-" * 50)

    conventions = NamingConventions()
    print("Context types:")
    for key, value in conventions.CONTEXT_TYPES.items():
        print(f"  {key} -> {value}")

    print("\nMetric patterns:")
    for key, pattern in conventions.METRIC_PATTERNS.items():
        print(f"  {key}: {pattern}")

    # Standards checking
    print("\n" + "-" * 50)
    print("STANDARDS CHECKING")
    print("-" * 50)

    checker = create_standard_checker()

    # Good context
    good_context = ContextResult(
        content="This is the context content.",
        sources=["doc_1", "doc_2"],
        metadata={"topic": "AI"},
        version="1.0.0",
        timestamp="2024-01-01T00:00:00",
        quality_score=0.85,
    )

    result = checker.check(good_context, "context")
    print(f"\nGood context check:")
    print(f"  Passed: {result.passed}")
    print(f"  Violations: {result.violations}")
    print(f"  Warnings: {result.warnings}")

    # Bad context
    bad_context = ContextResult(
        content="Missing sources and version",
        sources=[],
        metadata={},
        version="",
        timestamp="2024-01-01T00:00:00",
    )

    result = checker.check(bad_context, "context")
    print(f"\nBad context check:")
    print(f"  Passed: {result.passed}")
    print(f"  Violations: {result.violations}")
    print(f"  Warnings: {result.warnings}")

    print("\n" + "-" * 50)
    print("STANDARDS SYSTEM READY")
    print("-" * 50)
