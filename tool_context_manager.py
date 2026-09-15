"""
TOOL CONTEXT MANAGER
Four strategies for keeping tool context small (Lesson 6.2)

  1. Dynamic tool loading      -> load only relevant tool schemas
  2. Tool result compression   -> rule-based first, LLM upgrade as a drop-in swap
  3. Tool invocation context   -> keep the "why" + key finding, not raw data
  4. Error context handling    -> compact, actionable errors

Runs fully offline (rule-based). To see the LLM compression upgrade, pass an
Anthropic client to ToolContextManager(..., client=client) — see the note in
the DEMO section.

Run: python tool_context_manager.py
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~1.3 tokens per whitespace word)."""
    return int(len(text.split()) * 1.3)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    category: str

    def schema_tokens(self) -> int:
        """Estimate tokens used by this tool's schema."""
        schema_text = f"{self.name}: {self.description} {json.dumps(self.parameters)}"
        return estimate_tokens(schema_text)


# Sample tools
TOOLS = [
    Tool(
        "web_search",
        "Search the web for information",
        {"query": "string", "num_results": "int"},
        "research",
    ),
    Tool("read_webpage", "Read content from a URL", {"url": "string"}, "research"),
    Tool("database_query", "Query the database", {"sql": "string"}, "research"),
    Tool("calculator", "Perform calculations", {"expression": "string"}, "computation"),
    Tool("code_executor", "Execute Python code", {"code": "string"}, "computation"),
    Tool("file_reader", "Read a file", {"path": "string"}, "file_ops"),
    Tool(
        "file_writer",
        "Write to a file",
        {"path": "string", "content": "string"},
        "file_ops",
    ),
    Tool(
        "email_sender",
        "Send an email",
        {"to": "string", "subject": "string", "body": "string"},
        "communication",
    ),
]


# =============================================================================
# STRATEGY 1: DYNAMIC TOOL LOADER
# =============================================================================


class DynamicToolLoader:
    """Load only relevant tools based on the task."""

    def __init__(self, all_tools: List[Tool]):
        self.all_tools = all_tools
        self.tools_by_category: Dict[str, List[Tool]] = {}

        for tool in all_tools:
            self.tools_by_category.setdefault(tool.category, []).append(tool)

    def get_tools_for_task(self, task: str) -> List[Tool]:
        """Get relevant tools based on task analysis."""
        needed_categories = self._analyze_task(task)
        tools = []
        for category in needed_categories:
            tools.extend(self.tools_by_category.get(category, []))
        return tools

    def _analyze_task(self, task: str) -> List[str]:
        """Determine which tool categories are needed."""
        categories = []
        task_lower = task.lower()

        if any(w in task_lower for w in ["search", "find", "research", "look up"]):
            categories.append("research")
        if any(w in task_lower for w in ["calculate", "compute", "run", "execute"]):
            categories.append("computation")
        if any(w in task_lower for w in ["file", "read", "write", "save", "load"]):
            categories.append("file_ops")
        if any(w in task_lower for w in ["email", "send", "notify", "message"]):
            categories.append("communication")

        return categories or ["research"]  # Default to research

    def get_token_savings(self, task: str) -> dict:
        """Calculate token savings from dynamic loading."""
        all_tokens = sum(t.schema_tokens() for t in self.all_tools)
        selected = self.get_tools_for_task(task)
        selected_tokens = sum(t.schema_tokens() for t in selected)

        return {
            "all_tools": len(self.all_tools),
            "selected_tools": len(selected),
            "all_tokens": all_tokens,
            "selected_tokens": selected_tokens,
            "tokens_saved": all_tokens - selected_tokens,
            "savings_percent": round((1 - selected_tokens / all_tokens) * 100, 1),
        }


# =============================================================================
# STRATEGY 2: TOOL RESULT COMPRESSOR
# =============================================================================


class ToolResultCompressor:
    """Compress tool results to save context.

    Rule-based by default (instant, free, predictable). Pass an Anthropic
    `client` to upgrade web_search compression to LLM summarization — the
    public `compress()` signature never changes: it's a drop-in swap.
    """

    def __init__(self, max_result_tokens: int = 500, client: Any = None):
        self.max_result_tokens = max_result_tokens
        self.client = client  # optional; enables the LLM upgrade

    def compress(self, tool_name: str, result: str) -> str:
        """Compress a tool result to a manageable size."""
        if estimate_tokens(result) <= self.max_result_tokens:
            return result  # Small enough, keep as-is

        # Route to a tool-specific compressor
        if tool_name == "web_search":
            return self._compress_search(result)
        elif tool_name == "read_webpage":
            return self._compress_webpage(result)
        elif tool_name == "database_query":
            return self._compress_query(result)
        return self._generic_compress(result)

    def _compress_search(self, result: str) -> str:
        """Rule-based: keep first lines + URLs. LLM upgrade if a client is set."""
        if self.client is not None:
            return self._llm_summarize(result)

        lines = [ln[:200] for ln in result.split("\n") if ln.strip()]
        return "\n".join(lines[:15])

    def _llm_summarize(self, result: str) -> str:
        """LLM-powered upgrade: preserve meaning, not just the first lines."""
        prompt = (
            "Compress these search results to key findings only.\n"
            "Keep: main facts, URLs for the top 3 results, key quotes.\n"
            "Drop: redundant info, metadata, formatting.\n\n"
            f"Results:\n{result}\n\nCompressed (max 500 words):"
        )
        response = self.client.messages.create(
            model="claude-haiku-4-5",  # cheap + fast — ideal for summarization
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _compress_webpage(self, result: str) -> str:
        """Compress webpage content to main points."""
        if len(result) > 1500:
            return result[:1000] + "\n[...content truncated...]\n" + result[-500:]
        return result

    def _compress_query(self, result: str) -> str:
        """Compress database results."""
        lines = result.split("\n")
        if len(lines) > 20:
            return "\n".join(lines[:10]) + f"\n[...{len(lines) - 10} more rows...]"
        return result

    def _generic_compress(self, result: str) -> str:
        """Generic compression - truncate with a summary marker."""
        words = result.split()
        if len(words) > 300:
            return (
                " ".join(words[:150]) + f"\n[...truncated {len(words) - 150} words...]"
            )
        return result


# =============================================================================
# STRATEGY 3: TOOL INVOCATION CONTEXT
# =============================================================================


class ToolInvocationContext:
    """Format a tool call for context — the 'why' and key finding beat raw data."""

    def format_invocation(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        include_reasoning: bool = True,
    ) -> str:
        """Format a tool invocation for inclusion in context."""
        if not include_reasoning:
            truncated = result[:500] + ("..." if len(result) > 500 else "")
            return (
                f"## Tool: {tool_name}\n"
                f"Arguments: {json.dumps(arguments, indent=2)}\n"
                f"Result: {truncated}"
            )

        return (
            f"## Tool: {tool_name}\n"
            f"Why: Needed to {self._infer_purpose(tool_name, arguments)}\n"
            f"Arguments: {json.dumps(arguments)}\n"
            f"Key Finding: {self._extract_key_finding(result)}"
        )

    def _infer_purpose(self, tool_name: str, args: dict) -> str:
        """Explain why the tool was called."""
        purposes = {
            "web_search": f"find information about '{args.get('query', 'topic')}'",
            "read_webpage": f"read the page at '{args.get('url', 'a URL')}'",
            "file_reader": f"read contents of '{args.get('path', 'file')}'",
            "calculator": f"compute {args.get('expression', 'a calculation')}",
        }
        return purposes.get(tool_name, "complete the task")

    def _extract_key_finding(self, result: str) -> str:
        """Pull the single most useful line out of a result."""
        for line in result.split("\n"):
            line = line.strip()
            if line:
                return line[:120] + ("..." if len(line) > 120 else "")
        return "(no content)"


# =============================================================================
# STRATEGY 4: TOOL ERROR HANDLER
# =============================================================================


class ToolErrorHandler:
    """Turn verbose stack traces into compact, actionable error context."""

    def format_error(self, tool_name: str, error: Exception) -> str:
        """Create compact, actionable error context."""
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error)[:200],
            "recoverable": self._is_recoverable(error),
            "suggested_action": self._suggest_action(tool_name, error),
        }

        return (
            f"## Tool Error: {tool_name}\n"
            f"Error: {error_info['error_type']} - {error_info['message']}\n"
            f"Recoverable: {error_info['recoverable']}\n"
            f"Suggested: {error_info['suggested_action']}"
        )

    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if the error can be retried."""
        return isinstance(error, (TimeoutError, ConnectionError))

    def _suggest_action(self, tool_name: str, error: Exception) -> str:
        """Suggest the next action based on the error."""
        if isinstance(error, TimeoutError):
            return "Retry with longer timeout or simpler query"
        elif isinstance(error, PermissionError):
            return "Skip this file or request access"
        elif isinstance(error, ValueError):
            return "Validate input and retry with corrected arguments"
        return "Try an alternative approach"


# =============================================================================
# PUTTING IT ALL TOGETHER
# =============================================================================


class ToolContextManager:
    """Wire all four strategies into one context manager."""

    def __init__(self, tools: List[Tool], client: Any = None):
        self.loader = DynamicToolLoader(tools)
        # Pass a client to enable the LLM compression upgrade; omit for rule-based
        self.compressor = ToolResultCompressor(client=client)
        self.error_handler = ToolErrorHandler()
        self.invocation_formatter = ToolInvocationContext()
        self.invocation_history: List[dict] = []

    def get_tools(self, task: str, turn: int = 0) -> List[Tool]:
        """Get relevant tools, reducing the set as the task progresses."""
        tools = self.loader.get_tools_for_task(task)

        # Late in task? Reduce tool set
        if turn > 10:
            tools = self._filter_to_essential(tools)

        return tools

    def _filter_to_essential(self, tools: List[Tool]) -> List[Tool]:
        """Keep only the most essential tools late in a long task."""
        return tools[:3]

    def process_result(self, tool_name: str, arguments: dict, result: str) -> str:
        """Compress a result, then format it with reasoning for context."""
        compressed = self.compressor.compress(tool_name, result)
        formatted = self.invocation_formatter.format_invocation(
            tool_name, arguments, compressed
        )

        # Track for stats
        self.invocation_history.append(
            {
                "tool": tool_name,
                "original_tokens": estimate_tokens(result),
                "compressed_tokens": estimate_tokens(compressed),
            }
        )

        return formatted

    def handle_error(self, tool_name: str, error: Exception) -> str:
        """Handle a tool error with minimal, actionable context."""
        return self.error_handler.format_error(tool_name, error)

    def get_stats(self) -> dict:
        """Aggregate context-management statistics across all invocations."""
        if not self.invocation_history:
            return {"invocations": 0, "avg_savings_percent": 0}

        savings = [
            (1 - h["compressed_tokens"] / h["original_tokens"]) * 100
            for h in self.invocation_history
            if h["original_tokens"] > 0
        ]
        return {
            "invocations": len(self.invocation_history),
            "avg_savings_percent": (
                round(sum(savings) / len(savings), 1) if savings else 0
            ),
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TOOL CONTEXT MANAGER  -  4 strategies")
    print("=" * 60)

    # Rule-based by default. To enable the LLM compression upgrade (Strategy 2),
    # uncomment the two lines below and set ANTHROPIC_API_KEY:
    #   import anthropic
    #   manager = ToolContextManager(TOOLS, client=anthropic.Anthropic())
    manager = ToolContextManager(TOOLS)

    # ---- Strategy 1: Dynamic tool loading -----------------------------------
    print("\n" + "-" * 50)
    print("STRATEGY 1: DYNAMIC TOOL LOADING")
    print("-" * 50)

    tasks = [
        "Search for information about AI trends",
        "Calculate the ROI of this project",
        "Read the config file and update settings",
        "Send an email notification",
    ]

    for task in tasks:
        savings = manager.loader.get_token_savings(task)
        tools = manager.get_tools(task)
        print(f"\nTask: {task}")
        print(f"  Tools: {[t.name for t in tools]}")
        print(
            f"  Tokens: {savings['selected_tokens']} (saved {savings['tokens_saved']})"
        )
        print(f"  Savings: {savings['savings_percent']}%")

    # ---- Strategy 2: Result compression -------------------------------------
    print("\n" + "-" * 50)
    print("STRATEGY 2: RESULT COMPRESSION (rule-based)")
    print("-" * 50)

    large_search = """
Result 1: AI Trends 2025 - www.example.com/ai-trends
The artificial intelligence industry continues to grow rapidly...
Key findings include increased adoption of generative AI tools...
Market size expected to reach $500 billion by 2025...

Result 2: Machine Learning Advances - www.ml-news.com/advances
Recent breakthroughs in transformer architectures have enabled...
Modern LLMs represent significant advances in capability...
""" * 20  # Make it large enough to cross the 500-token threshold

    compressed = manager.compressor.compress("web_search", large_search)
    print(f"  Original:   {estimate_tokens(large_search)} tokens")
    print(f"  Compressed: {estimate_tokens(compressed)} tokens")
    mode = "LLM summarization" if manager.compressor.client else "rule-based truncation"
    print(f"  Mode:       {mode}")

    # ---- Strategy 3: Invocation context -------------------------------------
    print("\n" + "-" * 50)
    print("STRATEGY 3: INVOCATION CONTEXT")
    print("-" * 50)

    fmt = manager.invocation_formatter
    print("\nWithout reasoning (raw):")
    print(
        fmt.format_invocation(
            "web_search",
            {"query": "AI trends 2025"},
            large_search,
            include_reasoning=False,
        )[:200]
    )
    print("\nWith reasoning (why + key finding):")
    print(
        fmt.format_invocation("web_search", {"query": "AI trends 2025"}, large_search)
    )

    # ---- Strategy 4: Error handling -----------------------------------------
    print("\n" + "-" * 50)
    print("STRATEGY 4: ERROR HANDLING")
    print("-" * 50)

    for err in (
        TimeoutError("request timed out after 30s"),
        PermissionError("access denied: /etc/secret"),
        ValueError("invalid SQL syntax near 'SELCT'"),
    ):
        print("\n" + manager.handle_error("database_query", err))

    # ---- Integrated: process a result through the full pipeline --------------
    print("\n" + "-" * 50)
    print("INTEGRATED PIPELINE + STATS")
    print("-" * 50)

    print(
        manager.process_result("web_search", {"query": "AI trends 2025"}, large_search)
    )
    print("\nStats:", manager.get_stats())
