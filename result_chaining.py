"""
RESULT CHAINING
Chain tool results efficiently without context explosion

Run: python result_chaining.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
import json
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()  # Load environment variables from .env file


# =============================================================================
# CHAIN STEP MODEL
# =============================================================================


@dataclass
class ChainStep:
    step_id: str
    tool_name: str
    arguments: dict
    raw_result: str
    compressed_result: str
    key_data: dict
    tokens_original: int
    tokens_compressed: int


# =============================================================================
# CHAIN CONTEXT MANAGER
# =============================================================================


class ChainContextManager:
    """Manage context across a chain of tool calls."""

    def __init__(self, max_carry_forward: int = 500):
        self.max_carry_forward = max_carry_forward
        self.chain: List[ChainStep] = []
        self.step_counter = 0

    def add_step(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        extraction_template: str = None,
    ) -> ChainStep:
        """Add a step to the chain with compression."""
        self.step_counter += 1
        step_id = f"step_{self.step_counter}"

        # Calculate original tokens
        tokens_original = int(len(result.split()) * 1.3)

        # Extract key data
        key_data = self._extract_key_data(result, extraction_template)

        # Compress result
        compressed = self._compress_for_chain(result, tool_name)
        tokens_compressed = int(len(compressed.split()) * 1.3)

        step = ChainStep(
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            raw_result=result,
            compressed_result=compressed,
            key_data=key_data,
            tokens_original=tokens_original,
            tokens_compressed=tokens_compressed,
        )

        self.chain.append(step)
        return step

    def _extract_key_data(self, result: str, template: str = None) -> dict:
        """Extract structured data from result."""
        # In production, use LLM for extraction
        # Here we do simple pattern extraction

        key_data = {}

        # Extract URLs
        import re

        urls = re.findall(r"https?://[^\s]+", result)
        if urls:
            key_data["urls"] = urls[:5]

        # Extract numbers/stats
        numbers = re.findall(r"\$[\d,]+|\d+%|\d+\.\d+", result)
        if numbers:
            key_data["stats"] = numbers[:10]

        # Extract key phrases (simple heuristic)
        sentences = result.split(".")
        key_phrases = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        if key_phrases:
            key_data["key_points"] = key_phrases

        return key_data

    def _compress_for_chain(self, result: str, tool_name: str) -> str:
        """Compress result for chain context."""
        words = result.split()
        word_limit = self.max_carry_forward // 2  # Rough token estimate

        if len(words) <= word_limit:
            return result

        # Keep first part + key findings indicator
        compressed = " ".join(words[:word_limit])
        compressed += f"\n[...{len(words) - word_limit} words truncated...]"

        return compressed

    def get_chain_context(self) -> str:
        """Get compressed context from all chain steps."""
        if not self.chain:
            return ""

        lines = ["## Chain Progress"]

        for step in self.chain:
            lines.append(f"\n### {step.step_id}: {step.tool_name}")
            lines.append(f"Args: {json.dumps(step.arguments)}")
            lines.append(f"Result: {step.compressed_result}")

            if step.key_data:
                lines.append(f"Key Data: {json.dumps(step.key_data)}")

        return "\n".join(lines)

    def get_key_data(self, step_id: str = None) -> dict:
        """Get key data from specific step or all steps."""
        if step_id:
            for step in self.chain:
                if step.step_id == step_id:
                    return step.key_data
            return {}

        # Combine all key data
        combined = {}
        for step in self.chain:
            for key, value in step.key_data.items():
                if key not in combined:
                    combined[key] = []
                if isinstance(value, list):
                    combined[key].extend(value)
                else:
                    combined[key].append(value)

        return combined

    def get_stats(self) -> dict:
        """Get chain statistics."""
        if not self.chain:
            return {"steps": 0, "tokens_saved": 0}

        total_original = sum(s.tokens_original for s in self.chain)
        total_compressed = sum(s.tokens_compressed for s in self.chain)

        return {
            "steps": len(self.chain),
            "tokens_original": total_original,
            "tokens_compressed": total_compressed,
            "tokens_saved": total_original - total_compressed,
            "savings_percent": (
                round((1 - total_compressed / total_original) * 100, 1)
                if total_original > 0
                else 0
            ),
        }


# =============================================================================
# LLM-POWERED UPGRADE (drop-in swap)
# =============================================================================


class LLMChainContextManager(ChainContextManager):
    """Same chain manager, but extraction and compression use a model.

    Everything else (data model, add_step, the patterns below) is inherited
    unchanged. Only the two _extract/_compress methods are overridden.
    """

    def __init__(self, client, max_carry_forward: int = 500):
        super().__init__(max_carry_forward)
        self.client = client

    def _extract_key_data(self, result: str, template: str = None) -> dict:
        prompt = template or "Extract the key URLs, stats, and main points."
        response = self.client.messages.create(
            model="claude-haiku-4-5",  # cheap + fast — the right tier for extraction
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\nContent:\n{result}\n\nReturn JSON only:",
                },
                # Prefill an opening brace so the model must continue valid JSON.
                {"role": "assistant", "content": "{"},
            ],
        )
        # Re-add the prefilled "{" the model was forced to continue from.
        raw = "{" + response.content[0].text
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Best-effort JSON parse of a model response.

        Models often wrap JSON in ```json fences or add stray prose, which
        makes a naive json.loads() blow up on char 0. Strip fences, then fall
        back to slicing out the outermost {...} object.
        """
        text = text.strip()

        # Strip ```json ... ``` or ``` ... ``` fences if present.
        if text.startswith("```"):
            text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fall back to the first balanced-looking {...} block.
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        # Give the caller something usable instead of crashing the chain.
        return {"raw": text}

    def _compress_for_chain(self, result: str, tool_name: str) -> str:
        if int(len(result.split()) * 1.3) <= self.max_carry_forward:
            return result  # already small enough — don't pay for an API call

        response = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Compress this {tool_name} result to essentials only.\n"
                        "Keep: key facts, critical data, actionable items.\n"
                        "Drop: redundancy, verbose descriptions, metadata.\n\n"
                        f"Result:\n{result}\n\nCompressed (max 300 words):"
                    ),
                }
            ],
        )
        return response.content[0].text


# =============================================================================
# CHAIN PATTERNS
# =============================================================================


class LinearChain:
    """Execute a linear sequence of steps."""

    def __init__(self, chain_manager: ChainContextManager = None):
        self.chain_manager = chain_manager or ChainContextManager()

    def execute(self, steps: List[dict]) -> str:
        """
        Execute steps in sequence.

        Each step: {"tool": func, "args": dict, "name": str}
        """
        for step_def in steps:
            # Execute tool
            result = step_def["tool"](**step_def["args"])

            # Add to chain
            self.chain_manager.add_step(
                tool_name=step_def["name"], arguments=step_def["args"], result=result
            )

        return self.chain_manager.get_chain_context()


class FanOutChain:
    """Execute parallel steps then merge."""

    def __init__(self, chain_manager: ChainContextManager = None):
        self.chain_manager = chain_manager or ChainContextManager()

    def execute(
        self, initial_step: dict, parallel_steps: List[dict], merge_func: Callable
    ) -> str:
        """
        Execute initial step, then parallel steps, then merge.
        """
        # Initial step
        initial_result = initial_step["tool"](**initial_step["args"])
        self.chain_manager.add_step(
            tool_name=initial_step["name"],
            arguments=initial_step["args"],
            result=initial_result,
        )

        # Parallel steps
        parallel_results = []
        for step_def in parallel_steps:
            result = step_def["tool"](**step_def["args"])
            step = self.chain_manager.add_step(
                tool_name=step_def["name"], arguments=step_def["args"], result=result
            )
            parallel_results.append(step.compressed_result)

        # Merge
        merged = merge_func(parallel_results)
        self.chain_manager.add_step(
            tool_name="merge",
            arguments={"inputs": len(parallel_results)},
            result=merged,
        )

        return self.chain_manager.get_chain_context()


# =============================================================================
# DEMO
# =============================================================================


def mock_search(query: str) -> str:
    """Mock search tool."""
    return f"""
Search results for: {query}

1. Example Article - https://example.com/article1
   This is a comprehensive guide to {query}. It covers the basics
   and advanced topics. Market size is $50 billion. Growth rate 25%.

2. Research Paper - https://research.com/paper
   Academic analysis of {query}. Key findings include 85% adoption rate.
   The industry is expected to grow significantly.

3. News Article - https://news.com/latest
   Breaking news about {query}. Recent developments show promising trends.
   Experts predict continued innovation in this space.
""" * 3  # Make it larger


def mock_read(url: str) -> str:
    """Mock webpage reader."""
    return f"""
Content from {url}

This webpage contains detailed information about the topic.
The main points are:
1. Point one with supporting evidence and data
2. Point two with examples and case studies
3. Point three with recommendations

Statistics show that 72% of users prefer this approach.
Revenue in this sector reached $120 million last year.
The compound annual growth rate is approximately 18.5%.

Additional details include market analysis, competitive landscape,
and future projections based on current trends.
""" * 2


def mock_analyze(data: str) -> str:
    """Mock analysis tool."""
    return f"""
Analysis Results:

Summary: The data indicates strong growth potential.
Key Metrics:
- Growth rate: 18.5%
- Market size: $120M
- User satisfaction: 85%

Recommendations:
1. Focus on emerging markets
2. Invest in R&D
3. Expand partnerships

Confidence: High (based on 3 data sources)
"""


if __name__ == "__main__":
    print("=" * 60)
    print("RESULT CHAINING")
    print("=" * 60)

    # Demo 1: Linear chain
    print("\n" + "-" * 50)
    print("LINEAR CHAIN")
    print("-" * 50)

    chain = ChainContextManager()

    # Step 1: Search
    search_result = mock_search("AI market trends")
    step1 = chain.add_step(
        tool_name="web_search",
        arguments={"query": "AI market trends"},
        result=search_result,
    )
    print(f"\nStep 1: web_search")
    print(f"  Original: {step1.tokens_original} tokens")
    print(f"  Compressed: {step1.tokens_compressed} tokens")
    print(f"  Key data: {step1.key_data}")

    # Step 2: Read first URL
    if step1.key_data.get("urls"):
        read_result = mock_read(step1.key_data["urls"][0])
        step2 = chain.add_step(
            tool_name="read_webpage",
            arguments={"url": step1.key_data["urls"][0]},
            result=read_result,
        )
        print(f"\nStep 2: read_webpage")
        print(f"  Original: {step2.tokens_original} tokens")
        print(f"  Compressed: {step2.tokens_compressed} tokens")

    # Step 3: Analyze
    analyze_result = mock_analyze(chain.get_key_data())
    step3 = chain.add_step(
        tool_name="analyze", arguments={"data": "chain results"}, result=analyze_result
    )
    print(f"\nStep 3: analyze")
    print(f"  Original: {step3.tokens_original} tokens")
    print(f"  Compressed: {step3.tokens_compressed} tokens")

    # Show chain context
    print("\n" + "-" * 50)
    print("CHAIN CONTEXT (what goes to LLM)")
    print("-" * 50)
    context = chain.get_chain_context()
    print(context[:1500] + "..." if len(context) > 1500 else context)

    # Show stats
    print("\n" + "-" * 50)
    print("CHAIN STATISTICS")
    print("-" * 50)
    stats = chain.get_stats()
    print(f"Steps: {stats['steps']}")
    print(f"Original tokens: {stats['tokens_original']}")
    print(f"Compressed tokens: {stats['tokens_compressed']}")
    print(f"Tokens saved: {stats['tokens_saved']}")
    print(f"Savings: {stats['savings_percent']}%")

    # Demo 2: Show key data propagation
    print("\n" + "-" * 50)
    print("KEY DATA EXTRACTION")
    print("-" * 50)
    all_key_data = chain.get_key_data()
    print(f"All extracted data: {json.dumps(all_key_data, indent=2)}")

    # Demo 3: LLM-powered upgrade (drop-in swap) — only if an API key is set
    print("\n" + "-" * 50)
    print("LLM-POWERED UPGRADE (optional)")
    print("-" * 50)
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        llm_chain = LLMChainContextManager(client=anthropic.Anthropic())
        llm_step = llm_chain.add_step(
            tool_name="web_search",
            arguments={"query": "AI market trends"},
            result=mock_search("AI market trends"),
        )
        print(f"LLM compressed: {llm_step.tokens_compressed} tokens")
        print(f"LLM key data: {json.dumps(llm_step.key_data, indent=2)}")
    else:
        print("Set ANTHROPIC_API_KEY (and `pip install anthropic`) to run the")
        print("LLM-powered manager. The rule-based demo above needs neither.")
