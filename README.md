# RLM Implementation

A clean implementation of **Recursive Language Models (RLMs)** with **true context isolation**.

An RLM is an LLM that can write Python code, execute it in a local sandbox, observe the results, and — crucially — **call itself recursively** to solve sub-problems. Each recursive call gets a **fresh, empty context**, which prevents "Context Rot" (the degradation of LLM quality as conversation history grows).

Inspired by: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## Why RLM?

| Feature | Standard LLM | RLM |
|---|---|---|
| Code execution | ❌ Hallucinates results | ✅ Runs real Python |
| Complex tasks | Attempts everything in one go | Decomposes into sub-tasks |
| Context window | Fills up → "Context Rot" | Each sub-task gets fresh context |
| Self-correction | Limited | Sees errors, retries |

## Prerequisites
1. **Python 3.10+**
2. **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
3. **Groq API key** (free): [console.groq.com/keys](https://console.groq.com/keys)

## Installation

```bash
uv venv
uv pip install -e .
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

## Usage

```bash
uv run python demo.py
```

The root model receives a short `root_prompt`, while the full task payload is injected
into the sandbox as `context`. This keeps large prompts out of direct chat history.

### Model Configuration
```bash
# Groq (default) — fast inference, free tier available
RLM_MODEL=groq/llama-3.3-70b-versatile uv run python demo.py

# Local via Ollama
RLM_MODEL=ollama/llama3 uv run python demo.py

# Gemini (requires GEMINI_API_KEY env var)
RLM_MODEL=gemini/gemini-2.0-flash uv run python demo.py

# OpenAI (requires OPENAI_API_KEY env var)
RLM_MODEL=gpt-4o uv run python demo.py
```

## Project Structure
```
rlm-demo/
├── demo.py            # Entry point
├── pyproject.toml     # uv-managed dependencies
└── rlm/               # Core package
    ├── __init__.py
    ├── core.py         # RLM loop with true context isolation
    └── sandbox.py      # Local Python code executor
```

## How It Works
1. **User** sends a query.
2. **RLM** sends a compact root instruction to the LLM (not the full query text).
3. The full task text is available in Python as `context` inside the sandbox.
4. **RLM** asks the LLM for a plan (Thought) and code (Action).
5. **Sandbox** executes the code and returns stdout (Observation).
6. **RLM** feeds the observation back and repeats.
7. If the LLM writes `rlm_query("sub-task")` in its code, a **new RLM agent** is spawned with a **fresh context** to handle the sub-task. The parent only sees the final answer.
8. Recursion is capped via `max_depth` to prevent runaway child-agent loops.
9. When the LLM outputs `Final Answer:`, the loop ends.
