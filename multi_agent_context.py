"""
MULTI-AGENT CONTEXT SHARING
Context sharing between agents in a multi-agent system

Run: python multi_agent_context.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from queue import Queue
import json

# =============================================================================
# HANDOFF PROTOCOL
# =============================================================================


@dataclass
class AgentHandoff:
    """Context handoff between agents."""

    from_agent: str
    to_agent: str
    task: str
    context: str  # Compressed context
    key_data: dict  # Structured data
    confidence: float = 0.8


class HandoffManager:
    """Manage context handoffs between agents."""

    def __init__(self):
        self.handoff_history: List[AgentHandoff] = []

    def prepare_handoff(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        full_context: str,
        findings: dict,
    ) -> AgentHandoff:
        """Prepare a handoff to another agent."""

        # Compress context for receiving agent
        compressed = self._compress_for_agent(full_context, to_agent, task)

        # Filter findings to what receiving agent needs
        key_data = self._filter_for_agent(findings, to_agent)

        handoff = AgentHandoff(
            from_agent=from_agent,
            to_agent=to_agent,
            task=task,
            context=compressed,
            key_data=key_data,
        )

        self.handoff_history.append(handoff)
        return handoff

    def _compress_for_agent(self, context: str, to_agent: str, task: str) -> str:
        """Compress context based on receiving agent's needs."""
        # In production, use LLM for smart compression
        lines = context.split("\n")

        # Keep header and key findings
        compressed_lines = []
        for line in lines:
            # Keep lines with key data
            if any(
                marker in line.lower() for marker in ["finding", "result", "data", "##"]
            ):
                compressed_lines.append(line)

            # Stop at a reasonable length
            if len(compressed_lines) > 20:
                break

        if not compressed_lines:
            # Fallback: take first 500 chars
            return context[:500] + "..."

        return "\n".join(compressed_lines)

    def _filter_for_agent(self, findings: dict, to_agent: str) -> dict:
        """Filter findings based on what agent needs."""
        # Define what each agent type needs
        agent_needs = {
            "analyzer": ["data", "numbers", "sources"],
            "writer": ["conclusions", "key_points", "quotes"],
            "reviewer": ["draft", "requirements", "issues"],
            "researcher": ["queries", "sources", "gaps"],
        }

        needed = agent_needs.get(to_agent, list(findings.keys()))

        filtered = {}
        for key, value in findings.items():
            # Include if key matches needed categories
            if any(need in key.lower() for need in needed):
                filtered[key] = value

        # Return a shallow copy if nothing matched, so we never alias the live
        # results dict back into a result (which would create a circular ref).
        return filtered or dict(findings)


# =============================================================================
# SHARED MEMORY STORE
# =============================================================================


@dataclass
class MemoryEntry:
    """Entry in shared memory."""

    value: Any
    author: str
    visibility: str  # "all", "orchestrator", or specific agent_id
    timestamp: str
    version: int


class SharedMemoryStore:
    """Central memory accessible by all agents."""

    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
        self.access_log: List[dict] = []

    def write(self, agent_id: str, key: str, value: Any, visibility: str = "all"):
        """Write to shared memory."""
        version = 1
        if key in self.memories:
            version = self.memories[key].version + 1

        self.memories[key] = MemoryEntry(
            value=value,
            author=agent_id,
            visibility=visibility,
            timestamp=datetime.now().isoformat(),
            version=version,
        )

        self._log(agent_id, "write", key)

    def read(self, agent_id: str, key: str, default: Any = None) -> Any:
        """Read from shared memory (respecting visibility)."""
        if key not in self.memories:
            return default

        entry = self.memories[key]

        # Check visibility
        if self._can_access(agent_id, entry.visibility, entry.author):
            self._log(agent_id, "read", key)
            return entry.value

        return default

    def read_all_visible(self, agent_id: str) -> dict:
        """Get all memories visible to this agent."""
        visible = {}
        for key, entry in self.memories.items():
            if self._can_access(agent_id, entry.visibility, entry.author):
                visible[key] = entry.value
        return visible

    def _can_access(self, agent_id: str, visibility: str, author: str) -> bool:
        """Check if agent can access this memory."""
        if visibility == "all":
            return True
        if visibility == agent_id:
            return True
        if visibility == "orchestrator" and agent_id == "orchestrator":
            return True
        if agent_id == author:
            return True  # Author can always access
        return False

    def _log(self, agent_id: str, action: str, key: str):
        """Log access for debugging."""
        self.access_log.append(
            {
                "agent": agent_id,
                "action": action,
                "key": key,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "entries": len(self.memories),
            "accesses": len(self.access_log),
            "authors": len(set(e.author for e in self.memories.values())),
        }


# =============================================================================
# MESSAGE BUS
# =============================================================================


@dataclass
class AgentMessage:
    """Message between agents."""

    from_agent: str
    to_agent: str  # Or "broadcast"
    message_type: str  # "finding", "question", "status", "request"
    content: dict
    priority: int = 0


class AgentMessageBus:
    """Message bus for agent communication."""

    def __init__(self):
        self.queues: Dict[str, List[AgentMessage]] = {}
        self.message_history: List[AgentMessage] = []

    def register_agent(self, agent_id: str):
        """Register an agent to receive messages."""
        self.queues[agent_id] = []

    def send(self, message: AgentMessage):
        """Send message to agent or broadcast."""
        self.message_history.append(message)

        if message.to_agent == "broadcast":
            for agent_id in self.queues:
                if agent_id != message.from_agent:
                    self.queues[agent_id].append(message)
        else:
            if message.to_agent in self.queues:
                self.queues[message.to_agent].append(message)

    def receive(self, agent_id: str) -> Optional[AgentMessage]:
        """Get next message for agent."""
        if agent_id not in self.queues or not self.queues[agent_id]:
            return None

        # Return highest priority message
        self.queues[agent_id].sort(key=lambda m: m.priority, reverse=True)
        return self.queues[agent_id].pop(0)

    def receive_all(self, agent_id: str) -> List[AgentMessage]:
        """Get all messages for agent."""
        if agent_id not in self.queues:
            return []
        messages = self.queues[agent_id]
        self.queues[agent_id] = []
        return messages

    def broadcast_finding(self, from_agent: str, finding: str, data: dict = None):
        """Broadcast a finding to all agents."""
        self.send(
            AgentMessage(
                from_agent=from_agent,
                to_agent="broadcast",
                message_type="finding",
                content={"finding": finding, "data": data or {}},
            )
        )


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class SimpleOrchestrator:
    """Orchestrates multi-agent execution."""

    def __init__(self, workers: Dict[str, "SimpleWorker"]):
        self.workers = workers
        self.shared_memory = SharedMemoryStore()
        self.message_bus = AgentMessageBus()
        self.handoff_manager = HandoffManager()

        # Register all agents
        self.message_bus.register_agent("orchestrator")
        for worker_id in workers:
            self.message_bus.register_agent(worker_id)

    def execute_task(self, task: str, steps: List[dict]) -> dict:
        """
        Execute task using workers.

        steps: [{"id": "step1", "worker": "researcher", "task": "..."}]
        """
        results = {}

        for step in steps:
            worker_id = step["worker"]
            worker = self.workers.get(worker_id)

            if not worker:
                results[step["id"]] = {"error": f"Worker {worker_id} not found"}
                continue

            # Prepare handoff
            context = self._get_orchestrator_context()
            handoff = self.handoff_manager.prepare_handoff(
                from_agent="orchestrator",
                to_agent=worker_id,
                task=step["task"],
                full_context=context,
                findings=results,
            )

            # Execute
            result = worker.execute(handoff)
            results[step["id"]] = result

            # Store in shared memory
            self.shared_memory.write(worker_id, f"result_{step['id']}", result)

            # Broadcast finding
            self.message_bus.broadcast_finding(
                worker_id, f"Completed {step['task']}", result
            )

        return results

    def _get_orchestrator_context(self) -> str:
        """Get current orchestrator context."""
        visible = self.shared_memory.read_all_visible("orchestrator")
        return f"Current state: {json.dumps(visible, indent=2)}"


class SimpleWorker:
    """Simple worker agent."""

    def __init__(self, worker_id: str, specialty: str):
        self.worker_id = worker_id
        self.specialty = specialty

    def execute(self, handoff: AgentHandoff) -> dict:
        """Execute task from handoff."""
        # Simulate work based on specialty
        return {
            "worker": self.worker_id,
            "specialty": self.specialty,
            "task": handoff.task,
            "result": f"Completed: {handoff.task}",
            "data": handoff.key_data,
        }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-AGENT CONTEXT SHARING")
    print("=" * 60)

    # Create workers
    workers = {
        "researcher": SimpleWorker("researcher", "information gathering"),
        "analyzer": SimpleWorker("analyzer", "data analysis"),
        "writer": SimpleWorker("writer", "content creation"),
    }

    # Create orchestrator
    orchestrator = SimpleOrchestrator(workers)

    # Define task
    task = "Research AI market and create report"
    steps = [
        {"id": "research", "worker": "researcher", "task": "Search for AI market data"},
        {"id": "analyze", "worker": "analyzer", "task": "Analyze the findings"},
        {"id": "write", "worker": "writer", "task": "Write the report"},
    ]

    # Execute
    print("\n" + "-" * 50)
    print("EXECUTING MULTI-AGENT TASK")
    print("-" * 50)

    results = orchestrator.execute_task(task, steps)

    for step_id, result in results.items():
        print(f"\n{step_id}:")
        print(f"  Worker: {result['worker']}")
        print(f"  Result: {result['result']}")

    # Show shared memory
    print("\n" + "-" * 50)
    print("SHARED MEMORY STATE")
    print("-" * 50)

    all_memory = orchestrator.shared_memory.read_all_visible("orchestrator")
    for key, value in all_memory.items():
        print(f"{key}: {value}")

    # Show message history
    print("\n" + "-" * 50)
    print("MESSAGE BUS HISTORY")
    print("-" * 50)

    for msg in orchestrator.message_bus.message_history:
        print(
            f"{msg.from_agent} → {msg.to_agent}: [{msg.message_type}] {msg.content['finding']}"
        )

    # Show handoff history
    print("\n" + "-" * 50)
    print("HANDOFF HISTORY")
    print("-" * 50)

    for handoff in orchestrator.handoff_manager.handoff_history:
        print(f"{handoff.from_agent} → {handoff.to_agent}")
        print(f"  Task: {handoff.task}")
        print(f"  Context size: {len(handoff.context)} chars")
        print(f"  Key data: {list(handoff.key_data.keys())}")

    # Stats
    print("\n" + "-" * 50)
    print("STATISTICS")
    print("-" * 50)
    print(f"Memory stats: {orchestrator.shared_memory.get_stats()}")
    print(f"Handoffs: {len(orchestrator.handoff_manager.handoff_history)}")
    print(f"Messages: {len(orchestrator.message_bus.message_history)}")
