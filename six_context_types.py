"""
THE 6 CONTEXT TYPES
Examples of each context type for AI applications

Run: python six_context_types.py
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import json


@dataclass
class Example:
    user_input: str
    assistant_output: str


@dataclass
class RetrievedDoc:
    content: str
    source: str
    relevance_score: float


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class UserProfile:
    name: str
    expertise: str
    preferences: dict


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict


# =============================================================================
# TYPE 1: INSTRUCTIONAL CONTEXT
# =============================================================================

def build_instructional_context(
    role: str,
    behavior_rules: List[str],
    constraints: List[str]
) -> str:
    rules = "\n".join(f"- {rule}" for rule in behavior_rules)
    limits = "\n".join(f"- {c}" for c in constraints)

    return f"""## Role
{role}

## Behavior Rules
{rules}

## Constraints
{limits}"""


# =============================================================================
# TYPE 2: EXAMPLE CONTEXT
# =============================================================================

def build_example_context(examples: List[Example]) -> str:
    formatted = []
    for ex in examples:
        formatted.append(f"User: {ex.user_input}\nAssistant: {ex.assistant_output}")
    return "## Examples\n\n" + "\n\n".join(formatted)


# =============================================================================
# TYPE 3: KNOWLEDGE CONTEXT
# =============================================================================

def build_knowledge_context(docs: List[RetrievedDoc]) -> str:
    formatted = []
    for doc in docs:
        formatted.append(f"[Source: {doc.source}]\n{doc.content}")
    return "## Retrieved Documentation\n\n" + "\n\n".join(formatted)


# =============================================================================
# TYPE 4: MEMORY CONTEXT
# =============================================================================

def build_memory_context(
    history: List[ConversationTurn],
    profile: Optional[UserProfile] = None,
    max_turns: int = 5
) -> str:
    parts = []

    if history:
        recent = history[-max_turns:]
        history_text = "\n".join(f"{t.role}: {t.content}" for t in recent)
        parts.append(f"## Conversation History\n{history_text}")

    if profile:
        prefs = "\n".join(f"- {k}: {v}" for k, v in profile.preferences.items())
        parts.append(f"""## User Profile
- Name: {profile.name}
- Expertise: {profile.expertise}
{prefs}""")

    return "\n\n".join(parts)


# =============================================================================
# TYPE 5: TOOL CONTEXT
# =============================================================================

def build_tool_context(tools: List[ToolDefinition]) -> str:
    formatted = []
    for tool in tools:
        params = "\n".join(f"  - {k}: {v}" for k, v in tool.parameters.items())
        formatted.append(f"""### {tool.name}
{tool.description}
Parameters:
{params}""")
    return "## Available Tools\n\n" + "\n\n".join(formatted)


# =============================================================================
# TYPE 6: OUTPUT CONTEXT
# =============================================================================

def build_output_context(
    format_type: str,
    schema: Optional[dict] = None,
    constraints: List[str] = None
) -> str:
    parts = [f"## Output Requirements\n\nFormat: {format_type}"]

    if schema:
        parts.append(f"Schema:\n```json\n{json.dumps(schema, indent=2)}\n```")

    if constraints:
        c_text = "\n".join(f"- {c}" for c in constraints)
        parts.append(f"Constraints:\n{c_text}")

    return "\n\n".join(parts)


# =============================================================================
# FULL CONTEXT ASSEMBLY
# =============================================================================

def assemble_full_context(
    instructional: str,
    examples: str,
    knowledge: str,
    memory: str,
    tools: str,
    output: str,
    current_task: str
) -> str:
    parts = []
    if instructional:
        parts.append(instructional)
    if examples:
        parts.append(examples)
    if knowledge:
        parts.append(knowledge)
    if memory:
        parts.append(memory)
    if tools:
        parts.append(tools)
    if output:
        parts.append(output)
    parts.append(f"## Current Task\n{current_task}")
    return "\n\n---\n\n".join(parts)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("THE 6 CONTEXT TYPES")
    print("=" * 60)

    # Build each type
    instructional = build_instructional_context(
        role="You are a senior software engineer.",
        behavior_rules=["Be concise", "Include code examples"],
        constraints=["Max 300 words", "Python 3.10+ only"]
    )
    print("\n--- TYPE 1: INSTRUCTIONAL ---")
    print(instructional)

    examples = build_example_context([
        Example("How to read a file?", "Use `with open('file.txt') as f: data = f.read()`")
    ])
    print("\n--- TYPE 2: EXAMPLES ---")
    print(examples)

    knowledge = build_knowledge_context([
        RetrievedDoc("Python supports async file I/O.", "Python Docs", 0.95)
    ])
    print("\n--- TYPE 3: KNOWLEDGE ---")
    print(knowledge)

    memory = build_memory_context(
        history=[ConversationTurn("User", "I'm working with FastAPI")],
        profile=UserProfile("Alex", "Intermediate", {"framework": "FastAPI"})
    )
    print("\n--- TYPE 4: MEMORY ---")
    print(memory)

    tools = build_tool_context([
        ToolDefinition("run_code", "Execute Python", {"code": "string"})
    ])
    print("\n--- TYPE 5: TOOLS ---")
    print(tools)

    output = build_output_context(
        "JSON",
        {"answer": "string", "code": "string"},
        ["Include code example"]
    )
    print("\n--- TYPE 6: OUTPUT ---")
    print(output)

    # Full assembly
    full = assemble_full_context(
        instructional, examples, knowledge, memory, tools, output,
        "How do I read a CSV file?"
    )
    print("\n" + "=" * 60)
    print(f"FULL CONTEXT: {len(full)} chars (~{len(full)//4} tokens)")
