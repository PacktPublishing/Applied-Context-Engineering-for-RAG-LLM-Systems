"""
TESTING & VALIDATION
Context quality testing pyramid

Run: python testing_validation.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import json

# =============================================================================
# TEST TYPES
# =============================================================================


class TestLevel(Enum):
    UNIT = "unit"
    QUALITY = "quality"
    INTEGRATION = "integration"
    REGRESSION = "regression"


class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    name: str
    level: TestLevel
    status: TestStatus
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    name: str
    results: List[TestResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


# =============================================================================
# UNIT TESTS
# =============================================================================


class UnitTestRunner:
    """Run unit tests for context components."""

    def __init__(self):
        self.tests: List[Callable] = []

    def add_test(self, test_fn: Callable) -> None:
        self.tests.append(test_fn)

    def run(self) -> TestSuite:
        suite = TestSuite(name="unit_tests", started_at=datetime.now())

        for test_fn in self.tests:
            start = datetime.now()
            try:
                test_fn()
                status = TestStatus.PASSED
                message = "Test passed"
            except AssertionError as e:
                status = TestStatus.FAILED
                message = str(e)
            except Exception as e:
                status = TestStatus.ERROR
                message = f"Error: {str(e)}"

            duration = (datetime.now() - start).total_seconds() * 1000

            suite.add_result(
                TestResult(
                    name=test_fn.__name__,
                    level=TestLevel.UNIT,
                    status=status,
                    duration_ms=duration,
                    message=message,
                )
            )

        suite.completed_at = datetime.now()
        return suite


# =============================================================================
# QUALITY TESTS
# =============================================================================


@dataclass
class QualityThresholds:
    min_relevance: float = 0.7
    min_coverage: float = 0.8
    max_tokens: int = 50000
    max_latency_ms: float = 1000


class QualityTester:
    """Test context quality metrics."""

    def __init__(self, thresholds: QualityThresholds = None):
        self.thresholds = thresholds or QualityThresholds()

    def test_relevance(self, score: float) -> TestResult:
        passed = score >= self.thresholds.min_relevance
        return TestResult(
            name="relevance_check",
            level=TestLevel.QUALITY,
            status=TestStatus.PASSED if passed else TestStatus.FAILED,
            duration_ms=0,
            message=f"Relevance {score:.2f} vs threshold {self.thresholds.min_relevance}",
            details={"score": score, "threshold": self.thresholds.min_relevance},
        )

    def test_coverage(self, score: float) -> TestResult:
        passed = score >= self.thresholds.min_coverage
        return TestResult(
            name="coverage_check",
            level=TestLevel.QUALITY,
            status=TestStatus.PASSED if passed else TestStatus.FAILED,
            duration_ms=0,
            message=f"Coverage {score:.2f} vs threshold {self.thresholds.min_coverage}",
            details={"score": score, "threshold": self.thresholds.min_coverage},
        )

    def test_token_budget(self, tokens: int) -> TestResult:
        passed = tokens <= self.thresholds.max_tokens
        return TestResult(
            name="token_budget_check",
            level=TestLevel.QUALITY,
            status=TestStatus.PASSED if passed else TestStatus.FAILED,
            duration_ms=0,
            message=f"Tokens {tokens} vs max {self.thresholds.max_tokens}",
            details={"tokens": tokens, "max": self.thresholds.max_tokens},
        )

    def test_latency(self, latency_ms: float) -> TestResult:
        passed = latency_ms <= self.thresholds.max_latency_ms
        return TestResult(
            name="latency_check",
            level=TestLevel.QUALITY,
            status=TestStatus.PASSED if passed else TestStatus.FAILED,
            duration_ms=0,
            message=f"Latency {latency_ms:.1f}ms vs max {self.thresholds.max_latency_ms}ms",
            details={"latency": latency_ms, "max": self.thresholds.max_latency_ms},
        )

    def run_all(
        self, relevance: float, coverage: float, tokens: int, latency_ms: float
    ) -> TestSuite:
        suite = TestSuite(name="quality_tests", started_at=datetime.now())

        suite.add_result(self.test_relevance(relevance))
        suite.add_result(self.test_coverage(coverage))
        suite.add_result(self.test_token_budget(tokens))
        suite.add_result(self.test_latency(latency_ms))

        suite.completed_at = datetime.now()
        return suite


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@dataclass
class IntegrationTestCase:
    name: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    validators: List[Callable] = field(default_factory=list)


class IntegrationTester:
    """Test end-to-end context pipeline."""

    def __init__(self):
        self.test_cases: List[IntegrationTestCase] = []

    def add_case(self, case: IntegrationTestCase) -> None:
        self.test_cases.append(case)

    def run(self, pipeline_fn: Callable) -> TestSuite:
        suite = TestSuite(name="integration_tests", started_at=datetime.now())

        for case in self.test_cases:
            start = datetime.now()

            try:
                actual = pipeline_fn(case.input_data)

                # Run validators
                all_passed = True
                messages = []

                for validator in case.validators:
                    try:
                        validator(actual, case.expected_output)
                    except AssertionError as e:
                        all_passed = False
                        messages.append(str(e))

                status = TestStatus.PASSED if all_passed else TestStatus.FAILED
                message = "All validators passed" if all_passed else "; ".join(messages)

            except Exception as e:
                status = TestStatus.ERROR
                message = f"Pipeline error: {str(e)}"
                actual = {}

            duration = (datetime.now() - start).total_seconds() * 1000

            suite.add_result(
                TestResult(
                    name=case.name,
                    level=TestLevel.INTEGRATION,
                    status=status,
                    duration_ms=duration,
                    message=message,
                    details={"input": case.input_data, "output": actual},
                )
            )

        suite.completed_at = datetime.now()
        return suite


# =============================================================================
# REGRESSION TESTS
# =============================================================================


@dataclass
class BaselineSnapshot:
    query: str
    expected_docs: List[str]
    expected_quality: float
    created_at: datetime


class RegressionTestSuite:
    """Regression testing against baselines."""

    def __init__(self):
        self.baselines: Dict[str, BaselineSnapshot] = {}
        self.tolerance: float = 0.05

    def add_baseline(self, name: str, baseline: BaselineSnapshot) -> None:
        self.baselines[name] = baseline

    def run(self, retrieval_fn: Callable) -> TestSuite:
        suite = TestSuite(name="regression_tests", started_at=datetime.now())

        for name, baseline in self.baselines.items():
            start = datetime.now()

            try:
                result = retrieval_fn(baseline.query)
                actual_docs = [d.get("id") for d in result.get("documents", [])]
                actual_quality = result.get("quality", 0.0)

                # Check document overlap
                expected_set = set(baseline.expected_docs)
                actual_set = set(actual_docs[: len(baseline.expected_docs)])
                overlap = (
                    len(expected_set & actual_set) / len(expected_set)
                    if expected_set
                    else 1.0
                )

                # Check quality regression
                quality_diff = abs(actual_quality - baseline.expected_quality)
                quality_ok = quality_diff <= self.tolerance

                passed = overlap >= 0.8 and quality_ok
                status = TestStatus.PASSED if passed else TestStatus.FAILED

                message = f"Overlap: {overlap:.0%}, Quality diff: {quality_diff:.2f}"

            except Exception as e:
                status = TestStatus.ERROR
                message = f"Error: {str(e)}"
                overlap = 0
                quality_diff = 0

            duration = (datetime.now() - start).total_seconds() * 1000

            suite.add_result(
                TestResult(
                    name=f"regression_{name}",
                    level=TestLevel.REGRESSION,
                    status=status,
                    duration_ms=duration,
                    message=message,
                    details={
                        "baseline": name,
                        "overlap": overlap,
                        "quality_diff": quality_diff,
                    },
                )
            )

        suite.completed_at = datetime.now()
        return suite


# =============================================================================
# CONTEXT VALIDATOR
# =============================================================================


@dataclass
class ValidationRule:
    name: str
    check_fn: Callable[[Dict], bool]
    error_message: str


class ContextValidator:
    """Validate context before use."""

    def __init__(self):
        self.rules: List[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> None:
        self.rules.append(rule)

    def validate(self, context: Dict) -> TestSuite:
        suite = TestSuite(name="validation", started_at=datetime.now())

        for rule in self.rules:
            start = datetime.now()

            try:
                passed = rule.check_fn(context)
                status = TestStatus.PASSED if passed else TestStatus.FAILED
                message = "Valid" if passed else rule.error_message
            except Exception as e:
                status = TestStatus.ERROR
                message = f"Validation error: {str(e)}"

            duration = (datetime.now() - start).total_seconds() * 1000

            suite.add_result(
                TestResult(
                    name=rule.name,
                    level=TestLevel.QUALITY,
                    status=status,
                    duration_ms=duration,
                    message=message,
                )
            )

        suite.completed_at = datetime.now()
        return suite


def create_standard_validator() -> ContextValidator:
    """Create validator with standard rules."""
    validator = ContextValidator()

    validator.add_rule(
        ValidationRule(
            name="has_content",
            check_fn=lambda c: bool(c.get("content")),
            error_message="Context must have content",
        )
    )

    validator.add_rule(
        ValidationRule(
            name="has_sources",
            check_fn=lambda c: len(c.get("sources", [])) > 0,
            error_message="Context must have at least one source",
        )
    )

    validator.add_rule(
        ValidationRule(
            name="valid_token_count",
            check_fn=lambda c: 0 < c.get("token_count", 0) < 100000,
            error_message="Token count must be between 0 and 100000",
        )
    )

    return validator


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING & VALIDATION")
    print("=" * 60)

    # Unit tests
    print("\n" + "-" * 50)
    print("UNIT TESTS")
    print("-" * 50)

    runner = UnitTestRunner()

    def test_token_counting():
        text = "hello world"
        assert len(text.split()) == 2

    def test_relevance_calculation():
        score = 0.85
        assert score > 0.7

    def test_failing_assertion():
        assert 1 == 2, "Numbers don't match"

    runner.add_test(test_token_counting)
    runner.add_test(test_relevance_calculation)
    runner.add_test(test_failing_assertion)

    suite = runner.run()
    print(f"Unit tests: {suite.passed}/{suite.total} passed ({suite.pass_rate:.0%})")

    # Quality tests
    print("\n" + "-" * 50)
    print("QUALITY TESTS")
    print("-" * 50)

    tester = QualityTester(
        QualityThresholds(
            min_relevance=0.7, min_coverage=0.8, max_tokens=50000, max_latency_ms=500
        )
    )

    quality_suite = tester.run_all(
        relevance=0.85, coverage=0.90, tokens=25000, latency_ms=250
    )

    print(f"Quality tests: {quality_suite.passed}/{quality_suite.total} passed")
    for result in quality_suite.results:
        status = "✓" if result.status == TestStatus.PASSED else "✗"
        print(f"  {status} {result.name}: {result.message}")

    # Context validation
    print("\n" + "-" * 50)
    print("CONTEXT VALIDATION")
    print("-" * 50)

    validator = create_standard_validator()

    # Good context
    good_context = {
        "content": "This is the context content.",
        "sources": ["doc1", "doc2"],
        "token_count": 5000,
    }

    val_suite = validator.validate(good_context)
    print(f"Good context: {val_suite.passed}/{val_suite.total} rules passed")

    # Bad context
    bad_context = {"content": "", "sources": [], "token_count": 0}

    val_suite = validator.validate(bad_context)
    print(f"Bad context: {val_suite.passed}/{val_suite.total} rules passed")
    for result in val_suite.results:
        if result.status == TestStatus.FAILED:
            print(f"  ✗ {result.name}: {result.message}")

    # Integration tests
    print("\n" + "-" * 50)
    print("INTEGRATION TESTS")
    print("-" * 50)

    integration = IntegrationTester()

    integration.add_case(
        IntegrationTestCase(
            name="basic_retrieval",
            input_data={"query": "test query"},
            expected_output={"documents": [], "count": 0},
            validators=[lambda actual, expected: True],  # Simple pass
        )
    )

    def mock_pipeline(input_data):
        return {"documents": [], "count": 0}

    int_suite = integration.run(mock_pipeline)
    print(f"Integration: {int_suite.passed}/{int_suite.total} passed")

    # Regression tests
    print("\n" + "-" * 50)
    print("REGRESSION TESTS")
    print("-" * 50)

    regression = RegressionTestSuite()

    # Baseline that still holds: retrieval unchanged since last release
    regression.add_baseline(
        "search_ml",
        BaselineSnapshot(
            query="machine learning basics",
            expected_docs=["doc_1", "doc_3"],
            expected_quality=0.85,
            created_at=datetime.now(),
        ),
    )

    # Baseline that a change broke: used to return doc_2, now returns doc_5
    regression.add_baseline(
        "rank_python",
        BaselineSnapshot(
            query="python data tooling",
            expected_docs=["doc_2"],
            expected_quality=0.90,
            created_at=datetime.now(),
        ),
    )

    def mock_retrieval(query: str) -> Dict[str, Any]:
        # Simulates the CURRENT pipeline after a context change
        responses = {
            "machine learning basics": {
                "documents": [{"id": "doc_1"}, {"id": "doc_3"}],
                "quality": 0.86,
            },
            "python data tooling": {
                "documents": [{"id": "doc_5"}],  # regressed: was doc_2
                "quality": 0.71,
            },
        }
        return responses.get(query, {"documents": [], "quality": 0.0})

    reg_suite = regression.run(mock_retrieval)
    print(f"Regression: {reg_suite.passed}/{reg_suite.total} passed")
    for result in reg_suite.results:
        status = "✓" if result.status == TestStatus.PASSED else "✗"
        print(f"  {status} {result.name}: {result.message}")

    print("\n" + "-" * 50)
    print("TESTING PYRAMID COMPLETE")
    print("-" * 50)
