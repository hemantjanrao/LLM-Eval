# Getting Started

This guide walks you through the lab from zero to your first LLM-as-judge evaluation.

## Prerequisites

- Python 3.10 or newer (3.13 recommended)
- [Poetry](https://python-poetry.org/) for dependency management

## Install

```bash
git clone <your-repo-url>
cd LLM-Eval
poetry env use 3.13          # optional
poetry install --with dev,docs
```

Verify the deterministic test suite (no API key required):

```bash
poetry run pytest
```

You should see 4 passing tests covering the toy RAG pipeline, dataset, and local metrics.

## Your First Commands

### 1. Ask the toy RAG app a question

```bash
poetry run llm-eval-lab ask "What is DeepEval useful for?"
```

The toy app uses keyword retrieval and a deterministic answer generator — not a real LLM. That keeps behavior reproducible while you learn evaluation.

### 2. Inspect evaluation records

```bash
poetry run llm-eval-lab inspect-dataset
```

This runs all three dataset examples through the pipeline and shows:

- The user question
- The generated answer
- How many context chunks were retrieved

### 3. Score with local (API-free) metrics

```bash
poetry run llm-eval-lab score-local
```

Local metrics approximate the ideas behind RAG evaluation without calling an LLM. They are fast, free, and deterministic — ideal for CI and learning.

## Enable LLM-Based Metrics

DeepEval and Ragas use an evaluator LLM (here: OpenAI) to judge quality.

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

Smoke-test each framework:

```bash
poetry run llm-eval-lab deepeval-smoke
poetry run llm-eval-lab ragas-smoke
```

Run the full DeepEval test file:

```bash
poetry run deepeval test run evals/deepeval/test_rag_deepeval.py
```

## Browse the Docs Locally

```bash
poetry run mkdocs serve
```

Open `http://127.0.0.1:8000` in your browser.

## What to Read Next

1. [Concepts](concepts.md) — understand retrieval vs generation metrics
2. [Architecture](architecture.md) — how modules connect
3. [Learning Roadmap](learning-roadmap.md) — progressive skill path
4. [DeepEval](deepeval.md) and [Ragas](ragas.md) — framework-specific notes
