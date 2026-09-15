"""
COMPLETE AGENT CONTEXT SYSTEM
Full production agent with all context management features

Run: python complete_agent_system.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import anthropic
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re

from dotenv import load_dotenv

load_dotenv()


client = anthropic.Anthropic()


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    category: str


@dataclass
class ToolResult:
    tool_name: str
    arguments: dict
    raw_result: str
    compressed_result: str
    key_data: dict
    tokens_original: int
    tokens_compressed: int


@dataclass
class AgentTurn:
    turn_number: int
    action: str
    content: str
    summary: str


@dataclass
class AgentState:
    task: str
    status: str
    steps_completed: List[str] = field(default_factory=list)
    steps_pending: List[str] = field(default_factory=list)
    findings: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# DYNAMIC TOOL MANAGER
# =============================================================================


class DynamicToolManager:
    """Load and manage tools dynamically."""

    def __init__(self):
        self.all_tools = [
            Tool("web_search", "Search the web", {"query": "str"}, "research"),
            Tool("read_webpage", "Read a webpage", {"url": "str"}, "research"),
            Tool("analyze_data", "Analyze data", {"data": "str"}, "analysis"),
            Tool("summarize", "Summarize content", {"content": "str"}, "analysis"),
            Tool(
                "write_file",
                "Write to file",
                {"path": "str", "content": "str"},
                "output",
            ),
        ]

        self.by_category: Dict[str, List[Tool]] = {}
        for tool in self.all_tools:
            if tool.category not in self.by_category:
                self.by_category[tool.category] = []
            self.by_category[tool.category].append(tool)

    def get_tools_for_task(self, task: str, turn: int) -> List[Tool]:
        """Get relevant tools based on task and turn."""
        needed = self._analyze_intent(task)
        tools = []

        for cat in needed:
            tools.extend(self.by_category.get(cat, []))

        # Reduce tool set later in task
        if turn > 8:
            tools = [t for t in tools if t.category in ["analysis", "output"]]

        return tools

    def _analyze_intent(self, task: str) -> List[str]:
        """Determine needed tool categories."""
        categories = []
        task_lower = task.lower()

        if any(w in task_lower for w in ["search", "find", "research"]):
            categories.append("research")
        if any(w in task_lower for w in ["analyze", "compare", "evaluate"]):
            categories.append("analysis")
        if any(w in task_lower for w in ["write", "create", "generate"]):
            categories.append("output")

        return categories or ["research", "analysis"]

    def get_schemas(self, tools: List[Tool]) -> List[dict]:
        """Get tool schemas for API."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in t.parameters},
                    "required": list(t.parameters.keys()),
                },
            }
            for t in tools
        ]


# =============================================================================
# RESULT COMPRESSOR
# =============================================================================


class ResultCompressor:
    """Compress tool results for context efficiency."""

    def __init__(self, max_tokens: int = 500):
        self.max_tokens = max_tokens

    def compress(self, tool_name: str, result: str) -> ToolResult:
        """Compress a tool result."""
        tokens_original = int(len(result.split()) * 1.3)

        # Extract key data
        key_data = self._extract_key_data(result)

        # Compress based on tool type
        if tokens_original <= self.max_tokens:
            compressed = result
        elif tool_name == "web_search":
            compressed = self._compress_search(result)
        elif tool_name == "read_webpage":
            compressed = self._compress_webpage(result)
        else:
            compressed = self._generic_compress(result)

        tokens_compressed = int(len(compressed.split()) * 1.3)

        return ToolResult(
            tool_name=tool_name,
            arguments={},
            raw_result=result,
            compressed_result=compressed,
            key_data=key_data,
            tokens_original=tokens_original,
            tokens_compressed=tokens_compressed,
        )

    def _extract_key_data(self, result: str) -> dict:
        """Extract structured data."""
        data = {}

        # URLs
        urls = re.findall(r"https?://[^\s]+", result)
        if urls:
            data["urls"] = urls[:5]

        # Numbers/stats
        stats = re.findall(r"\$[\d,]+|\d+%|\d+\.\d+[KMB]?", result)
        if stats:
            data["stats"] = stats[:10]

        return data

    def _compress_search(self, result: str) -> str:
        """Compress search results."""
        lines = result.split("\n")
        return "\n".join(lines[:15])

    def _compress_webpage(self, result: str) -> str:
        """Compress webpage content."""
        if len(result) > 1500:
            return result[:1000] + "\n[...]\n" + result[-500:]
        return result

    def _generic_compress(self, result: str) -> str:
        """Generic compression."""
        words = result.split()
        return " ".join(words[:300]) + "..."


# =============================================================================
# AGENT MEMORY
# =============================================================================


class AgentMemory:
    """Multi-turn memory with compression."""

    def __init__(self, max_turns_detail: int = 5):
        self.max_turns_detail = max_turns_detail
        self.turns: List[AgentTurn] = []
        self.state = AgentState(task="", status="waiting")
        self.turn_counter = 0
        self.summaries: List[str] = []

    def set_task(self, task: str):
        """Initialize with task."""
        self.state.task = task
        self.state.status = "executing"

    def start_turn(self) -> int:
        """Start a new turn."""
        self.turn_counter += 1
        return self.turn_counter

    def add_turn(self, action: str, content: str):
        """Record a turn."""
        summary = content[:100] + "..." if len(content) > 100 else content

        turn = AgentTurn(
            turn_number=self.turn_counter,
            action=action,
            content=content,
            summary=summary,
        )
        self.turns.append(turn)

        # Compress if needed
        if len(self.turns) > self.max_turns_detail * 2:
            self._compress_old()

    def _compress_old(self):
        """Compress older turns."""
        keep = self.max_turns_detail
        old = self.turns[:-keep]

        # Create summary
        summary = "; ".join([t.summary for t in old])
        self.summaries.append(summary)

        # Keep only recent
        self.turns = self.turns[-keep:]

    def add_finding(self, key: str, value: str):
        """Add a key finding."""
        self.state.findings[key] = value

    def set_complete(self):
        """Mark as complete."""
        self.state.status = "complete"

    def is_complete(self) -> bool:
        return self.state.status == "complete"

    def get_context(self) -> str:
        """Build context for LLM."""
        sections = []

        # Task
        sections.append(f"## Task\n{self.state.task}")

        # Status
        sections.append(f"## Status\n{self.state.status}")

        # Summaries of old turns
        if self.summaries:
            sections.append(f"## Earlier\n{'; '.join(self.summaries[-2:])}")

        # Recent turns
        if self.turns:
            recent = "\n".join(
                [f"- Turn {t.turn_number}: {t.summary}" for t in self.turns[-5:]]
            )
            sections.append(f"## Recent Actions\n{recent}")

        # Findings
        if self.state.findings:
            findings = "\n".join(
                [f"- {k}: {v}" for k, v in self.state.findings.items()]
            )
            sections.append(f"## Key Findings\n{findings}")

        return "\n\n".join(sections)


# =============================================================================
# CONTEXT ISOLATOR
# =============================================================================


class ContextIsolator:
    """Sanitize external content."""

    def __init__(self, max_external_tokens: int = 1000):
        self.max_external_tokens = max_external_tokens
        self.blocked_count = 0

    def sanitize(self, content: str, source: str) -> str:
        """Sanitize external content."""
        # Remove injection patterns
        patterns = [
            r"(?i)ignore.*instruction",
            r"(?i)you are now",
            r"(?i)system\s*:",
            r"(?i)disregard",
        ]

        result = content
        for pattern in patterns:
            matches = re.findall(pattern, result)
            self.blocked_count += len(matches)
            result = re.sub(pattern, "[blocked]", result)

        # Truncate
        words = result.split()
        if len(words) > self.max_external_tokens:
            result = " ".join(words[: self.max_external_tokens]) + "..."

        return f"[Source: {source}]\n{result}"


# =============================================================================
# MOCK TOOLS (for demo without real API calls)
# =============================================================================


def mock_web_search(query: str) -> str:
    return f"""
Search results for: {query}

1. AI Coding Assistants - www.example.com/ai-coding
   Comprehensive guide to AI coding tools. Market growing at 25%.
   Key players: GitHub Copilot, Cursor, Codeium.

2. Developer Tools Report - www.devtools.io/report
   Industry analysis shows $5B market size.
   85% of developers now use AI assistance.

3. Comparison Guide - www.techreview.com/compare
   Cursor leads in IDE integration.
   Copilot has largest user base at 1.5M users.
""" * 40  # Simulate a large, real search-results payload


def mock_read_webpage(url: str) -> str:
    return f"""
Content from {url}


The AI coding assistant market has seen explosive growth.
Key metrics:
- Market size: $5 billion
- Growth rate: 25% CAGR
- User adoption: 85% of professional developers

Top players:
1. GitHub Copilot - 1.5M users, $10/month
2. Cursor - 500K users, $20/month
3. Codeium - 300K users, free tier available

Features comparison shows Cursor leading in accuracy
while Copilot has best IDE integration coverage.
""" * 40  # Simulate a large, real webpage payload


def mock_analyze(data: str) -> str:
    return """
Analysis Results:

Market Leader: GitHub Copilot (by user base)
Fastest Growing: Cursor (by growth rate)
Best Value: Codeium (free tier)

Recommendation: For enterprise, Copilot.
For individuals, Cursor offers best features/price.

Confidence: High (based on 3 sources)
"""


MOCK_TOOLS = {
    "web_search": mock_web_search,
    "read_webpage": mock_read_webpage,
    "analyze_data": mock_analyze,
}


# =============================================================================
# COMPLETE AGENT
# =============================================================================


class ResearchAgent:
    """Complete research agent with context management."""

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        if not use_mock:
            self.client = anthropic.Anthropic()

        self.tool_manager = DynamicToolManager()
        self.compressor = ResultCompressor()
        self.memory = AgentMemory()
        # High cap: the isolator sanitizes + marks source; routine size
        # reduction is the compressor's job, so we don't pre-truncate here.
        self.isolator = ContextIsolator(max_external_tokens=4000)

        # Stats
        self.total_tokens_original = 0
        self.total_tokens_compressed = 0

    def run(self, task: str, max_turns: int = 10) -> str:
        """Execute research task."""
        self.memory.set_task(task)

        print(f"\n{'='*60}")
        print(f"RESEARCH AGENT")
        print(f"{'='*60}")
        print(f"\nTask: {task}\n")

        # Simulate agent loop
        for _ in range(max_turns):
            turn = self.memory.start_turn()

            # Get tools
            tools = self.tool_manager.get_tools_for_task(task, turn)
            print(f"Turn {turn}:")
            print(f"  Tools available: {[t.name for t in tools]}")

            # Simulate decision (in real agent, call LLM)
            action = self._decide_action(turn)

            if action["type"] == "tool_call":
                result = self._execute_tool(action["tool"], action["args"])
                print(f"  Action: {action['tool']}")
                print(
                    f"  Tokens: {result.tokens_original} → {result.tokens_compressed} "
                    f"({100 - int(result.tokens_compressed/result.tokens_original*100)}% saved)"
                )

                # Record turn
                self.memory.add_turn(
                    "tool_call",
                    f"{action['tool']}: {result.compressed_result[:100]}...",
                )

                # Track stats
                self.total_tokens_original += result.tokens_original
                self.total_tokens_compressed += result.tokens_compressed

            elif action["type"] == "complete":
                print(f"  Action: Complete")
                self.memory.set_complete()
                break

        # Generate final report
        return self._generate_report()

    def _decide_action(self, turn: int) -> dict:
        """Simulate agent decisions."""
        # Scripted sequence for demo
        if turn == 1:
            return {
                "type": "tool_call",
                "tool": "web_search",
                "args": {"query": "AI coding assistants comparison"},
            }
        elif turn == 2:
            return {
                "type": "tool_call",
                "tool": "read_webpage",
                "args": {"url": "www.techreview.com/compare"},
            }
        elif turn == 3:
            return {
                "type": "tool_call",
                "tool": "analyze_data",
                "args": {"data": "collected findings"},
            }
        else:
            return {"type": "complete"}

    def _execute_tool(self, tool_name: str, args: dict) -> ToolResult:
        """Execute tool and compress result."""
        # Get raw result
        if self.use_mock and tool_name in MOCK_TOOLS:
            raw = MOCK_TOOLS[tool_name](args.get("query", args.get("url", "")))
        else:
            raw = f"[Would execute {tool_name} with {args}]"

        # Sanitize (treat as external)
        sanitized = self.isolator.sanitize(raw, f"tool:{tool_name}")

        # Compress
        result = self.compressor.compress(tool_name, sanitized)
        result.arguments = args

        # Store finding if significant
        if result.key_data:
            self.memory.add_finding(tool_name, json.dumps(result.key_data)[:200])

        return result

    def _generate_report(self) -> str:
        """Generate final report."""
        context = self.memory.get_context()

        report = f"""
{'='*60}
FINAL REPORT
{'='*60}

Based on research conducted:

{context}

{'='*60}
CONTEXT EFFICIENCY REPORT
{'='*60}
Total tool results: {self.total_tokens_original} tokens
Compressed to: {self.total_tokens_compressed} tokens
Savings: {100 - int(self.total_tokens_compressed/max(1,self.total_tokens_original)*100)}%

Memory turns: {self.memory.turn_counter}
Turns in detail: {len(self.memory.turns)}
Compressed summaries: {len(self.memory.summaries)}

Injections blocked: {self.isolator.blocked_count}
"""
        return report


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    agent = ResearchAgent(use_mock=True)
    report = agent.run(
        "Research the top 3 competitors in the AI coding assistant space"
    )
    print(report)
