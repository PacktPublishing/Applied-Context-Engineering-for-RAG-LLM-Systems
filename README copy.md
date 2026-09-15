<!-- @format -->

# Context Engineering for AI Developers

## What You'll Learn

- ✅ The difference between prompt engineering and context engineering
- ✅ Why well-engineered context outperforms model-switching every time
- ✅ The 5 context layers every production AI app needs
- ✅ Real Python code — not toy examples
- ✅ Error handling patterns for LangChain + Claude + Pinecone stacks

---

## Tech Stack

| Tool                  | Version | Purpose               |
| --------------------- | ------- | --------------------- |
| `anthropic`           | 0.104.1 | Claude API SDK        |
| `langchain`           | 1.3.1   | LLM orchestration     |
| `langchain-anthropic` | 1.4.3   | LangChain x Claude    |
| `pinecone`            | 9.0.1   | Vector database       |
| `python-dotenv`       | latest  | Environment variables |

---

## Project Structure

```
context-engineering-course/
├── README.md
├── pyproject.toml              # uv-managed dependencies
├── .env.example                # Environment variable template
├── .gitignore
├── src/
│   └── context_engineering/
│       ├── __init__.py
│       ├── 01_prompt_engineering.py     # The baseline — what prompt engineering is
│       ├── 02_context_engineering.py    # Context engineering intro + comparison
│       ├── 03_five_context_layers.py    # The 5 layers framework with full code
│       ├── 04_rag_context_pipeline.py   # RAG + context engineering combined
│       ├── 05_error_handling.py         # Production error handling patterns
│       └── utils/
│           ├── __init__.py
│           └── context_builder.py       # Reusable context builder class
├── examples/
│   └── demo_comparison.py              # Side-by-side demo from the video
└── assets/
    └── context_layers_diagram.md       # Visual reference for the 5 layers
```

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key → [console.anthropic.com](https://console.anthropic.com)
- Pinecone API key → [pinecone.io](https://pinecone.io) (optional, for RAG examples)

### Install

```bash
# Clone the repo
git clone https://github.com/paulodichone/context-engineering-course.git
cd context-engineering-course

# Install dependencies with uv
uv sync

# Copy and fill in your API keys
cp .env.example .env
```

### Run the demo

```bash
# See the side-by-side comparison from the video
uv run python examples/demo_comparison.py

# Run individual modules
uv run python src/context_engineering/01_prompt_engineering.py
uv run python src/context_engineering/03_five_context_layers.py
```

---

## The 5 Context Layers (Quick Reference)

```
Layer 1 — System Identity    → WHO the AI is + its constraints
Layer 2 — User Context       → WHO is asking + their history
Layer 3 — Retrieved Knowledge → WHAT it needs to know (RAG)
Layer 4 — Conversation Memory → WHAT was said before (last 5-10 turns)
Layer 5 — State & Tools      → WHAT it can do + current system state
```

---
