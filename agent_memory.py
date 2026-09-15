"""
AGENT MEMORY
Multi-turn memory for agent state management

Run: python agent_memory.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()  # Load environment variables from .env file


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class AgentTurn:
    turn_number: int
    action_type: str  # "tool_call", "response", "think"
    action_details: str
    summary: str
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentState:
    task: str
    status: str  # "planning", "executing", "waiting", "complete"
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    key_findings: Dict[str, str] = field(default_factory=dict)
    current_step: Optional[str] = None


# =============================================================================
# AGENT MEMORY SYSTEM
# =============================================================================


class AgentMemory:
    """Complete memory system for multi-turn agents."""

    def __init__(self, max_turns_in_detail: int = 5):
        self.max_turns_in_detail = max_turns_in_detail
        self.turns: List[AgentTurn] = []
        self.state = AgentState(task="", status="waiting")
        self.turn_counter = 0
        self.summary_archive: List[str] = []

    def set_task(self, task: str, steps: Optional[List[str]] = None):
        """Initialize memory with task."""
        self.state.task = task
        self.state.status = "planning"
        self.state.pending_steps = steps or []

    def start_turn(self) -> int:
        """Begin a new turn."""
        self.turn_counter += 1
        self.state.status = "executing"
        return self.turn_counter

    def record_action(
        self, action_type: str, action_details: str, importance: float = 0.5
    ):
        """Record an action taken during this turn."""
        summary = self._create_summary(action_type, action_details)

        turn = AgentTurn(
            turn_number=self.turn_counter,
            action_type=action_type,
            action_details=action_details,
            summary=summary,
            importance=importance,
        )

        self.turns.append(turn)

        # Once the detailed window overflows, archive the oldest turns so we
        # never hold more than `max_turns_in_detail` turns at full fidelity.
        if len(self.turns) > self.max_turns_in_detail:
            self._compress_old_turns()

    def _create_summary(self, action_type: str, details: str) -> str:
        """Create a concise summary of the action."""
        # In production, use LLM for summarization
        if action_type == "tool_call":
            # Extract tool name and key result
            lines = details.split("\n")
            first_line = lines[0] if lines else details[:100]
            return f"Called tool: {first_line}"

        elif action_type == "think":
            return f"Reasoning: {details[:100]}..."

        elif action_type == "response":
            return f"Responded: {details[:100]}..."

        return details[:100]

    def _compress_old_turns(self):
        """Compress older turns into summary."""
        if len(self.turns) <= self.max_turns_in_detail:
            return

        # Keep recent turns in detail
        recent = self.turns[-self.max_turns_in_detail :]
        old = self.turns[: -self.max_turns_in_detail]

        # Create summary of old turns
        summaries = [t.summary for t in old]
        combined_summary = "; ".join(summaries)

        # Archive and clear
        self.summary_archive.append(combined_summary)
        self.turns = recent

    def complete_step(self, step: str, outcome: Optional[str] = None):
        """Mark a step as complete."""
        if step in self.state.pending_steps:
            self.state.pending_steps.remove(step)
        self.state.completed_steps.append(step)

        if outcome:
            self.state.key_findings[step] = outcome

        self.state.current_step = None

    def start_step(self, step: str):
        """Mark step as in progress."""
        self.state.current_step = step

    def set_complete(self):
        """Mark task as complete."""
        self.state.status = "complete"

    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.state.status == "complete"

    def get_context(self, max_tokens: int = 3000) -> str:
        """Build memory context for LLM.

        `max_tokens` is the budget this context should fit within. Compression
        (archiving old turns) is what keeps us under it; a production system
        would also trim sections here and count tokens with a real tokenizer.
        """
        sections = []

        # Task and state (always include)
        sections.append(self._format_state())

        # Archived summaries (brief)
        if self.summary_archive:
            archive_text = " | ".join(self.summary_archive[-3:])  # Last 3 archives
            sections.append(f"## Earlier History\n{archive_text}")

        # Recent turns (detailed)
        if self.turns:
            sections.append(self._format_recent_turns())

        # Key findings (always include)
        if self.state.key_findings:
            sections.append(self._format_findings())

        return "\n\n".join(sections)

    def _format_state(self) -> str:
        """Format current state."""
        return f"""## Task
{self.state.task}

## Progress
Status: {self.state.status}
Current: {self.state.current_step or 'Deciding next action'}
Completed: {len(self.state.completed_steps)} steps
Remaining: {len(self.state.pending_steps)} steps"""

    def _format_recent_turns(self) -> str:
        """Format recent turns."""
        lines = ["## Recent Actions"]

        for turn in self.turns:
            importance_marker = "!" if turn.importance >= 0.8 else ""
            lines.append(
                f"- [Turn {turn.turn_number}]{importance_marker} {turn.summary}"
            )

        return "\n".join(lines)

    def _format_findings(self) -> str:
        """Format key findings."""
        lines = ["## Key Findings"]

        for step, finding in self.state.key_findings.items():
            lines.append(f"- {step}: {finding}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_turns": self.turn_counter,
            "turns_in_memory": len(self.turns),
            "archives": len(self.summary_archive),
            "steps_completed": len(self.state.completed_steps),
            "steps_pending": len(self.state.pending_steps),
            "findings": len(self.state.key_findings),
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AGENT MEMORY SYSTEM")
    print("=" * 60)

    memory = AgentMemory(max_turns_in_detail=3)

    # Initialize task
    memory.set_task(
        task="Research competitors in the AI assistant market",
        steps=[
            "Search for competitors",
            "Analyze top 3 companies",
            "Compare features",
            "Write summary",
        ],
    )

    # Simulate several turns
    print("\n" + "-" * 50)
    print("SIMULATING AGENT TURNS")
    print("-" * 50)

    # Turn 1: Search
    memory.start_turn()
    memory.start_step("Search for competitors")
    memory.record_action(
        "tool_call",
        "web_search: 'AI assistant companies 2026'\nFound: OpenAI, Anthropic, Google DeepMind, Microsoft, Cursor",
        importance=0.9,
    )
    memory.complete_step("Search for competitors", "Found 5 main competitors")
    print(f"Turn 1: Searched for competitors")

    # Turn 2: Analyze OpenAI
    memory.start_turn()
    memory.start_step("Analyze top 3 companies")
    memory.record_action(
        "tool_call",
        "read_webpage: openai.com\nGPT-5 is their flagship, ChatGPT has ~800M weekly users",
        importance=0.8,
    )
    print(f"Turn 2: Analyzed OpenAI")

    # Turn 3: Analyze Anthropic
    memory.start_turn()
    memory.record_action(
        "tool_call",
        "read_webpage: anthropic.com\nClaude Opus 4 leads on coding + safety, enterprise focus",
        importance=0.8,
    )
    print(f"Turn 3: Analyzed Anthropic")

    # Turn 4: Analyze Google
    memory.start_turn()
    memory.record_action(
        "tool_call",
        "read_webpage: deepmind.google\nGemini 3 powers Google products, strong research",
        importance=0.7,
    )
    memory.complete_step(
        "Analyze top 3 companies", "Analyzed OpenAI, Anthropic, Google"
    )
    print(f"Turn 4: Analyzed Google")

    # Turn 5: Compare
    memory.start_turn()
    memory.start_step("Compare features")
    memory.record_action(
        "think",
        "Comparing: OpenAI leads in users, Anthropic in coding/safety, Google in integration",
        importance=0.85,
    )
    memory.complete_step(
        "Compare features", "OpenAI=scale, Anthropic=coding/safety, Google=integration"
    )
    print(f"Turn 5: Compared features")

    # Turn 6: Write summary
    memory.start_turn()
    memory.start_step("Write summary")
    memory.record_action(
        "response",
        "The AI assistant market is dominated by three players with distinct strategies...",
        importance=0.9,
    )
    memory.complete_step("Write summary")
    memory.set_complete()
    print(f"Turn 6: Wrote summary")

    # Show memory state
    print("\n" + "-" * 50)
    print("MEMORY CONTEXT (for LLM)")
    print("-" * 50)
    print(memory.get_context())

    # Show stats
    print("\n" + "-" * 50)
    print("MEMORY STATISTICS")
    print("-" * 50)
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Show compression effect
    print("\n" + "-" * 50)
    print("COMPRESSION EFFECT")
    print("-" * 50)
    print(f"Total turns recorded: {stats['total_turns']}")
    print(f"Turns in detail: {stats['turns_in_memory']}")
    print(f"Archives created: {stats['archives']}")
    if memory.summary_archive:
        print(f"Archive content: {memory.summary_archive[0][:100]}...")
