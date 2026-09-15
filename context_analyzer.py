"""
CONTEXT PERFORMANCE ANALYZER
============================

THE LESSON THIS SCRIPT PROVES:

    Context quality beats context quantity.
    More tokens often HURT both accuracy AND your wallet.

How it proves it:
  - Asks the same question against 4 different contexts:
      Minimal / Structured / Noisy / Bloated
  - Runs each context N times (default 3) to expose variance
  - Grades answers with an LLM judge (semantic match, not string match)
  - Detects hallucinations (facts in the answer not in the context)
  - Reports cost per call AND cost per CORRECT answer
  - Prints a final takeaway students can actually quote

Run:    uv run context_analyzer.py
Env:    ANTHROPIC_API_KEY (required)
        ANSWER_MODEL, JUDGE_MODEL, RUNS_PER_CONTEXT,
        PRICE_IN_PER_M, PRICE_OUT_PER_M (all optional)
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

CLIENT = anthropic.Anthropic()

# Model used by the "student" to answer the question.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-4-5")

# Model used as the grader. Same model is fine; a stronger model reduces bias.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-5")

# Pricing per 1M tokens for the ANSWER model (USD). Override for other models.
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "3.00"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "15.00"))

# Repeats per context — exposes variance instead of pretending one call is truth.
RUNS_PER_CONTEXT = int(os.getenv("RUNS_PER_CONTEXT", "3"))


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    accuracy: float  # 0..1 — semantic match to expected answer
    hallucination: float  # 0..1 — share of answer claims not supported by context
    reasoning: str


@dataclass
class RunResult:
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    verdict: JudgeVerdict


@dataclass
class ContextReport:
    label: str
    context_chars: int
    runs: list[RunResult]

    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    avg_accuracy: float = 0.0
    accuracy_stdev: float = 0.0
    avg_hallucination: float = 0.0
    signal_density: float = 0.0  # relevant sentences / total sentences
    avg_input_cost: float = 0.0
    avg_output_cost: float = 0.0
    avg_total_cost: float = 0.0
    cost_per_correct_answer: float = 0.0  # the real efficiency metric
    representative_answer: str = ""


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------


def answer_question(context: str, question: str) -> tuple[str, int, int, float]:
    """Ask the student model to answer using only the supplied context."""
    start = time.time()
    resp = CLIENT.messages.create(
        model=ANSWER_MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using ONLY the context above. Be concise. "
                    "If the answer is not in the context, say so."
                ),
            }
        ],
    )
    latency_ms = (time.time() - start) * 1000
    return (
        resp.content[0].text,
        resp.usage.input_tokens,
        resp.usage.output_tokens,
        latency_ms,
    )


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def judge_answer(
    answer: str, question: str, expected: str, context: str
) -> JudgeVerdict:
    """LLM-as-judge: semantic accuracy and hallucination, returned as JSON."""
    prompt = (
        "You are an impartial grader.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED ANSWER (ground truth):\n{expected}\n\n"
        f"CONTEXT GIVEN TO THE STUDENT MODEL:\n{context}\n\n"
        f"STUDENT ANSWER:\n{answer}\n\n"
        "Grade on TWO axes and return ONLY valid JSON, no prose:\n"
        "{\n"
        '  "accuracy": <0.0-1.0 — does the student answer convey the same meaning '
        "as the expected answer? Semantic equivalence is fully accurate; do not "
        "require identical wording.>,\n"
        '  "hallucination": <0.0-1.0 — share of factual claims in the student '
        "answer that are NOT supported by the context.>,\n"
        '  "reasoning": "<one short sentence>"\n'
        "}\n"
    )
    resp = CLIENT.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _JSON_FENCE.sub("", resp.content[0].text.strip()).strip()
    try:
        data = json.loads(raw)
        return JudgeVerdict(
            accuracy=max(0.0, min(1.0, float(data.get("accuracy", 0)))),
            hallucination=max(0.0, min(1.0, float(data.get("hallucination", 0)))),
            reasoning=str(data.get("reasoning", "")).strip(),
        )
    except (json.JSONDecodeError, ValueError):
        return JudgeVerdict(
            accuracy=0.0,
            hallucination=1.0,
            reasoning=f"judge returned unparseable output: {raw[:80]}",
        )


def signal_density(context: str, keywords: list[str]) -> float:
    """Fraction of sentences in the context that mention any expected keyword.

    A deterministic proxy for 'how much of this context is actually on-topic'.
    Not perfect — but defensible, reproducible, and free.
    """
    sentences = [s for s in re.split(r"[.!?\n]+", context) if s.strip()]
    if not sentences:
        return 0.0
    keywords_lc = [k.lower() for k in keywords]
    relevant = sum(1 for s in sentences if any(k in s.lower() for k in keywords_lc))
    return relevant / len(sentences)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


def evaluate_context(
    context: str,
    question: str,
    expected: str,
    keywords: list[str],
    label: str,
    n_runs: int = RUNS_PER_CONTEXT,
) -> ContextReport:
    runs: list[RunResult] = []
    for _ in range(n_runs):
        answer, in_tok, out_tok, latency_ms = answer_question(context, question)
        verdict = judge_answer(answer, question, expected, context)
        runs.append(RunResult(answer, in_tok, out_tok, latency_ms, verdict))

    r = ContextReport(label=label, context_chars=len(context), runs=runs)
    r.avg_input_tokens = statistics.mean(x.input_tokens for x in runs)
    r.avg_output_tokens = statistics.mean(x.output_tokens for x in runs)
    r.avg_latency_ms = statistics.mean(x.latency_ms for x in runs)
    r.avg_accuracy = statistics.mean(x.verdict.accuracy for x in runs)
    r.avg_hallucination = statistics.mean(x.verdict.hallucination for x in runs)
    r.accuracy_stdev = (
        statistics.stdev(x.verdict.accuracy for x in runs) if len(runs) > 1 else 0.0
    )
    r.signal_density = signal_density(context, keywords)
    r.avg_input_cost = (r.avg_input_tokens / 1_000_000) * PRICE_IN_PER_M
    r.avg_output_cost = (r.avg_output_tokens / 1_000_000) * PRICE_OUT_PER_M
    r.avg_total_cost = r.avg_input_cost + r.avg_output_cost
    r.cost_per_correct_answer = (
        r.avg_total_cost / r.avg_accuracy if r.avg_accuracy > 0 else float("inf")
    )
    representative = min(runs, key=lambda x: abs(x.verdict.accuracy - r.avg_accuracy))
    r.representative_answer = representative.answer
    return r


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def bar(value: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return "█" * filled + "░" * (width - filled)


def status_for(accuracy: float) -> str:
    if accuracy >= 0.85:
        return "✓ GOOD"
    if accuracy >= 0.55:
        return "⚠ PARTIAL"
    return "✗ POOR"


def print_dashboard(r: ContextReport) -> None:
    line = "═" * 60
    print(f"\n{line}\n  {r.label}\n{line}")
    print(
        f"\n  Signal density:      [{bar(r.signal_density)}] {r.signal_density*100:5.0f}%"
    )
    print(
        f"  Accuracy (semantic): [{bar(r.avg_accuracy)}] {r.avg_accuracy*100:5.0f}% "
        f"±{r.accuracy_stdev*100:.0f}%  {status_for(r.avg_accuracy)}"
    )
    print(
        f"  Hallucination:       [{bar(r.avg_hallucination)}] {r.avg_hallucination*100:5.0f}%"
    )
    print()
    print(f"  Context size:        {r.context_chars:,} chars")
    print(f"  Avg input tokens:    {r.avg_input_tokens:,.0f}")
    print(f"  Avg output tokens:   {r.avg_output_tokens:,.0f}")
    print(f"  Avg latency:         {r.avg_latency_ms:,.0f} ms")
    print(
        f"  Cost / call:         ${r.avg_total_cost:.5f}  "
        f"(in ${r.avg_input_cost:.5f} + out ${r.avg_output_cost:.5f})"
    )
    cpca = (
        f"${r.cost_per_correct_answer:.5f}"
        if r.cost_per_correct_answer != float("inf")
        else "∞  (no correct answers)"
    )
    print(f"  Cost / correct ans:  {cpca}")
    print("\n  Sample answer:")
    sample = r.representative_answer.strip()
    if len(sample) > 280:
        sample = sample[:280] + "..."
    for chunk in sample.splitlines():
        print(f"    {chunk}")


def print_comparison(reports: list[ContextReport]) -> None:
    print("\n" + "=" * 84)
    print("  COMPARISON")
    print("=" * 84)
    print(
        f"\n  {'Context':<14} {'Tokens':>8} {'Signal':>8} "
        f"{'Accuracy':>11} {'Halluc':>8} {'$/call':>10} {'$/correct':>12}"
    )
    print("  " + "-" * 82)

    best_tokens = min(r.avg_input_tokens for r in reports)
    best_signal = max(r.signal_density for r in reports)
    best_acc = max(r.avg_accuracy for r in reports)
    best_hall = min(r.avg_hallucination for r in reports)
    best_call = min(r.avg_total_cost for r in reports)
    best_correct = min(r.cost_per_correct_answer for r in reports)

    def mark(val: float, best: float) -> str:
        return "★" if val == best else " "

    for r in reports:
        cpca = (
            f"${r.cost_per_correct_answer:.4f}"
            if r.cost_per_correct_answer != float("inf")
            else "n/a"
        )
        print(
            f"  {r.label:<14} "
            f"{r.avg_input_tokens:>7,.0f}{mark(r.avg_input_tokens, best_tokens)} "
            f"{r.signal_density*100:>6.0f}%{mark(r.signal_density, best_signal)} "
            f"{r.avg_accuracy*100:>9.0f}%{mark(r.avg_accuracy, best_acc)} "
            f"{r.avg_hallucination*100:>6.0f}%{mark(r.avg_hallucination, best_hall)} "
            f"${r.avg_total_cost:>8.5f}{mark(r.avg_total_cost, best_call)} "
            f"{cpca:>10}{mark(r.cost_per_correct_answer, best_correct)}"
        )
    print("\n  ★ = best in column")


def print_lesson(reports: list[ContextReport]) -> None:
    print("\n" + "=" * 84)
    print("  THE LESSON")
    print("=" * 84)

    most_accurate = max(reports, key=lambda r: r.avg_accuracy)
    biggest = max(reports, key=lambda r: r.avg_input_tokens)
    smallest = min(reports, key=lambda r: r.avg_input_tokens)
    finite = [r for r in reports if r.cost_per_correct_answer != float("inf")]
    most_efficient = (
        min(finite, key=lambda r: r.cost_per_correct_answer) if finite else None
    )

    token_blowup = biggest.avg_input_tokens / max(1, smallest.avg_input_tokens)
    cost_blowup = biggest.avg_total_cost / max(1e-9, smallest.avg_total_cost)

    print(
        f"\n  • Most accurate context : '{most_accurate.label}' "
        f"({most_accurate.avg_accuracy*100:.0f}% semantic accuracy)"
    )
    if most_efficient:
        print(
            f"  • Best $/correct answer : '{most_efficient.label}' "
            f"(${most_efficient.cost_per_correct_answer:.5f})"
        )
    print(
        f"  • '{biggest.label}' uses {token_blowup:.1f}× the tokens of "
        f"'{smallest.label}' and costs {cost_blowup:.1f}× more per call —"
    )
    if biggest is not most_accurate:
        print(f"    and it is NOT the most accurate. Bigger ≠ better.")
    else:
        print(f"    accuracy parity at best — the extra tokens bought nothing.")

    print("\n  What students should take away:")
    print("    1. Tokens are not free. Every irrelevant sentence costs money on")
    print("       every call, forever.")
    print("    2. Signal density (relevant sentences / total) predicts accuracy")
    print("       better than raw size does.")
    print("    3. Structure beats bulk — clear headings let the model land on the")
    print("       right span without dragging in noise or hallucinating around it.")
    print("    4. Measure cost-per-CORRECT-answer, not cost-per-call. A cheap call")
    print("       that's wrong is infinitely expensive.")
    print("    5. Context engineering = curation, not accumulation.")


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------


def run_demo() -> None:
    print("=" * 84)
    print("  CONTEXT PERFORMANCE ANALYZER")
    print("=" * 84)
    print(f"  Answer model     : {ANSWER_MODEL}")
    print(f"  Judge model      : {JUDGE_MODEL}")
    print(f"  Runs per context : {RUNS_PER_CONTEXT}")
    print(
        f"  Pricing          : ${PRICE_IN_PER_M:.2f}/M in, ${PRICE_OUT_PER_M:.2f}/M out"
    )

    question = "What is the refund policy for software purchases?"
    expected = (
        "Software purchases can be refunded within 7 days, but only if the "
        "software has not been activated or used."
    )
    keywords = ["refund", "software", "7 day", "unused", "activate"]

    print(f"\n  Question : {question}")
    print(f"  Expected : {expected}")

    contexts = {
        "Minimal": (
            "## Refund Policy\n"
            "- Software: 7-day refund if unused\n"
            "- Hardware: 30-day refund\n"
            "- Services: Non-refundable\n"
        ),
        "Structured": (
            "## Purchase Policies\n\n"
            "### Software Purchases\n"
            "Software purchases can be refunded within 7 days if the software "
            "has not been activated or used. Once activated, software is "
            "non-refundable.\n\n"
            "### Hardware Purchases\n"
            "Hardware can be returned within 30 days in its original packaging.\n\n"
            "### Service Purchases\n"
            "All service purchases are final and non-refundable.\n"
        ),
        "Noisy": (
            "Welcome to TechCorp! We've been serving customers since 1995.\n"
            "Our mission is to provide excellent technology solutions worldwide.\n\n"
            + ("About our team: we have 500 employees across 12 countries. " * 8)
            + "\n\nSomewhere in this document is our refund policy: software "
            "purchases can be refunded within 7 days if the software has not "
            "been activated or used.\n\n"
            + ("Our values include integrity, innovation, and customer focus. " * 8)
            + "\n"
        ),
        "Bloated": (
            ("Company history and background dating back to our 1995 founding. " * 30)
            + "\n\n"
            + ("Our leadership team bios, awards, and industry achievements. " * 20)
            + "\n\n"
            + ("Office locations across North America, Europe, and Asia. " * 15)
            + "\n\n"
            "Refund information: software — 7 days, unused only.\n\n"
            + (
                "Our future product roadmap, partnership plans, and investor updates. "
                * 30
            )
            + "\n"
        ),
    }

    reports: list[ContextReport] = []
    for label, ctx in contexts.items():
        print(
            f"\n  Evaluating: {label:<11} "
            f"({len(ctx):>5,} chars × {RUNS_PER_CONTEXT} runs)...",
            end=" ",
            flush=True,
        )
        reports.append(evaluate_context(ctx, question, expected, keywords, label))
        print("done.")

    for r in reports:
        print_dashboard(r)

    print_comparison(reports)
    print_lesson(reports)


if __name__ == "__main__":
    run_demo()
