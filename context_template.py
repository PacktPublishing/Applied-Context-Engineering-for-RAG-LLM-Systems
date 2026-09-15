"""
CONTEXT TEMPLATES FOR COMMON PATTERNS
Reusable templates for AI applications

5 templates covering the most common AI patterns:
    1. Customer Service Bot
    2. RAG Q&A System
    3. Code Assistant
    4. AI Agent
    5. Content Generator

Run: python context_template.py
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

# =============================================================================
# TEMPLATE 1: CUSTOMER SERVICE BOT
# =============================================================================


@dataclass
class CustomerProfile:
    name: str
    tier: str
    recent_issues: List[str]


def customer_service_template(
    company_name: str,
    tone: str,
    retrieved_policies: str,
    customer: CustomerProfile,
    conversation_history: str,
    question: str,
) -> str:
    """Template for customer service bots"""

    return f"""## Role
You are a {company_name} customer service agent.
Tone: {tone}

## Knowledge Base
{retrieved_policies}

## Customer Profile
- Name: {customer.name}
- Tier: {customer.tier}
- Previous Issues: {', '.join(customer.recent_issues) if customer.recent_issues else 'None'}

## Conversation History
{conversation_history if conversation_history else 'New conversation'}

## Guidelines
- Answer based ONLY on the knowledge base
- If unsure, offer to escalate to human agent
- Always confirm you've addressed the question
- For {customer.tier} customers, prioritize their satisfaction

## Response Format
1. Acknowledge the question
2. Provide the answer
3. Ask if they need anything else

## Customer Question
{question}"""


# =============================================================================
# TEMPLATE 2: RAG Q&A SYSTEM
# =============================================================================


@dataclass
class RetrievedDocument:
    content: str
    source: str
    score: float


def rag_qa_template(
    documents: List[RetrievedDocument], question: str, include_confidence: bool = True
) -> str:
    """Template for RAG-based Q&A systems"""

    docs_section = []
    for i, doc in enumerate(documents, 1):
        docs_section.append(f"""### Document {i}
Source: {doc.source}
Relevance: {doc.score:.2f}

{doc.content}""")

    docs_text = "\n\n".join(docs_section)

    confidence_section = (
        """
## Confidence
Rate your confidence: HIGH (answer clearly in docs), MEDIUM (partially covered), LOW (mostly inference)"""
        if include_confidence
        else ""
    )

    return f"""## Task
Answer the question based ONLY on the provided documents.
If the answer isn't in the documents, say "I don't have information about that in my knowledge base."

## Retrieved Documents

{docs_text}

## Important
- Cite sources using [Source: X] format
- Don't make up information not present in documents
- Be concise and direct
{confidence_section}

## Question
{question}

## Answer
Provide your answer, then list sources used."""


# =============================================================================
# TEMPLATE 3: CODE ASSISTANT
# =============================================================================


def code_assistant_template(
    language: str,
    framework: Optional[str],
    style_guide: str,
    relevant_code: str,
    style_examples: str,
    constraints: List[str],
    task: str,
) -> str:
    """Template for coding assistants"""

    framework_section = f"\nFramework: {framework}" if framework else ""
    constraints_text = "\n".join(f"- {c}" for c in constraints)

    return f"""## Role
You are an expert {language} developer.
Follow {style_guide} conventions.

## Project Context{framework_section}

## Relevant Existing Code
```{language}
{relevant_code}
```

## Style Examples (match this style)
{style_examples}

## Constraints
{constraints_text}

## Task
{task}

## Output
Provide:
1. The code solution (in a code block)
2. Brief explanation of the approach
3. Any assumptions made"""


# =============================================================================
# TEMPLATE 4: AI AGENT
# =============================================================================


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, str]
    use_cases: str


def agent_template(
    agent_name: str,
    agent_role: str,
    primary_goal: str,
    tools: List[ToolDefinition],
    action_history: List[str],
    current_state: str,
    current_task: str,
) -> str:
    """Template for autonomous agents with tools"""

    tools_section = []
    for tool in tools:
        params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
        tools_section.append(f"""### {tool.name}
{tool.description}
Parameters: {params}
Use when: {tool.use_cases}""")

    tools_text = "\n\n".join(tools_section)
    history_text = (
        "\n".join(f"- {a}" for a in action_history[-5:])
        if action_history
        else "No previous actions"
    )

    return f"""## Agent Identity
You are {agent_name}, a {agent_role}.
Your goal: {primary_goal}

## Available Tools
{tools_text}

## Execution Rules
1. Think step-by-step before acting
2. Use tools when they help achieve the goal
3. Verify results before proceeding
4. If stuck, explain what's blocking you
5. When goal is achieved, respond with DONE

## Current State
Previous actions:
{history_text}

Current observations: {current_state}

## Task
{current_task}

## Next Action
Decide your next action. Format:
THOUGHT: [Your reasoning about what to do next]
ACTION: [tool_name] or RESPOND or DONE
PARAMETERS: [if using tool, provide parameters as JSON]"""


# =============================================================================
# TEMPLATE 5: CONTENT GENERATOR
# =============================================================================


def content_generator_template(
    content_type: str,
    target_audience: str,
    expertise_level: str,
    tone: str,
    voice: str,
    key_points: List[str],
    reference_material: str,
    word_count: int,
    format_requirements: str,
    things_to_avoid: List[str],
    content_request: str,
) -> str:
    """Template for content generation (emails, blog posts, docs)"""

    points_text = "\n".join(f"- {p}" for p in key_points)
    avoid_text = "\n".join(f"- {a}" for a in things_to_avoid)

    return f"""## Content Type
{content_type}

## Audience
{target_audience}
Expertise level: {expertise_level}

## Tone & Style
Tone: {tone}
Voice: {voice}

## Key Points to Cover
{points_text}

## Reference Material
{reference_material}

## Constraints
- Length: approximately {word_count} words
- Format: {format_requirements}

## Avoid
{avoid_text}

## Generate
{content_request}"""


# =============================================================================
# DEMO: Render each template so students see the output
# =============================================================================


def demo_templates() -> None:
    """Render all 5 templates with realistic inputs and print the results."""

    print("=" * 60)
    print("CONTEXT TEMPLATES DEMO")
    print("=" * 60)

    # ---- TEMPLATE 1: Customer Service ----
    print("\n--- TEMPLATE 1: CUSTOMER SERVICE ---\n")
    cs_context = customer_service_template(
        company_name="TechCorp",
        tone="friendly and professional",
        retrieved_policies=(
            "Return policy: 30 days for most items. Electronics: 15 days."
        ),
        customer=CustomerProfile("Sarah", "Premium", ["shipping delay"]),
        conversation_history="User: Hi, I need help\nAssistant: Hello! How can I help?",
        question="What's your return policy for laptops?",
    )
    print(cs_context)
    print(f"\n[Prompt size: {len(cs_context)} chars (~{len(cs_context)//4} tokens)]")

    # ---- TEMPLATE 2: RAG Q&A ----
    print("\n\n--- TEMPLATE 2: RAG Q&A ---\n")
    rag_context = rag_qa_template(
        documents=[
            RetrievedDocument(
                "API rate limit is 100 requests per minute.", "API Docs", 0.95
            ),
            RetrievedDocument(
                "Exceeding limits returns 429 error.", "Error Guide", 0.87
            ),
        ],
        question="What happens if I exceed the rate limit?",
    )
    print(rag_context)
    print(f"\n[Prompt size: {len(rag_context)} chars (~{len(rag_context)//4} tokens)]")

    # ---- TEMPLATE 3: Code Assistant ----
    print("\n\n--- TEMPLATE 3: CODE ASSISTANT ---\n")
    code_context = code_assistant_template(
        language="Python",
        framework="FastAPI",
        style_guide="PEP 8",
        relevant_code="def get_user(id: int): ...",
        style_examples="async def endpoint(): return {'status': 'ok'}",
        constraints=["Use async/await", "Type hints required"],
        task="Create an endpoint to update user profile",
    )
    print(code_context)
    print(
        f"\n[Prompt size: {len(code_context)} chars (~{len(code_context)//4} tokens)]"
    )

    # ---- TEMPLATE 4: AI Agent ----
    print("\n\n--- TEMPLATE 4: AI AGENT ---\n")
    agent_context = agent_template(
        agent_name="ResearchBot",
        agent_role="research assistant",
        primary_goal="Find and summarize information on topics",
        tools=[
            ToolDefinition(
                "search", "Search the web", {"query": "string"}, "Finding information"
            ),
            ToolDefinition(
                "summarize", "Summarize text", {"text": "string"}, "Condensing content"
            ),
        ],
        action_history=["Searched for 'AI trends 2026'"],
        current_state="Found 5 relevant articles",
        current_task="Summarize the key AI trends",
    )
    print(agent_context)
    print(
        f"\n[Prompt size: {len(agent_context)} chars (~{len(agent_context)//4} tokens)]"
    )

    # ---- TEMPLATE 5: Content Generator ----
    print("\n\n--- TEMPLATE 5: CONTENT GENERATOR ---\n")
    content_context = content_generator_template(
        content_type="Blog post",
        target_audience="Software developers",
        expertise_level="Intermediate",
        tone="Technical but approachable",
        voice="Second person (you)",
        key_points=["Benefits of context engineering", "Common mistakes"],
        reference_material="Context engineering improves AI accuracy by 40%...",
        word_count=500,
        format_requirements="Use headers, include code examples",
        things_to_avoid=["Jargon without explanation", "Marketing speak"],
        content_request="Write a blog post about context engineering basics",
    )
    print(content_context)
    print(
        f"\n[Prompt size: {len(content_context)} chars (~{len(content_context)//4} tokens)]"
    )

    # ---- Wrap up ----
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print(
        "Templates turn context engineering into a repeatable function call.\n"
        "Same structure, different inputs -> consistent, production-grade prompts.\n"
        "Customize them: add/remove sections to fit your domain."
    )


if __name__ == "__main__":
    demo_templates()
