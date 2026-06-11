# Getting Started

This guide walks you through the lab from zero to your first structured evaluation run and optional LLM-as-judge metrics.

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

You should see 19 passing tests covering the toy RAG pipeline, JSONL dataset loading, config, runner, local metrics, and regression gates.

Optional: install pre-commit hooks for local linting before you push:

```bash
poetry run pre-commit install
```

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

This loads examples from `datasets/default.jsonl`, runs them through the pipeline, and shows:

- The user question
- The generated answer
- How many context chunks were retrieved

### 3. Score with local (API-free) metrics

```bash
poetry run llm-eval-lab score-local
```

Local metrics approximate the ideas behind RAG evaluation without calling an LLM. They are fast, free, and deterministic — ideal for CI and learning.

### 4. Run the full evaluation suite

```bash
poetry run llm-eval-lab run-eval
```

This is the main production entry point. It reads `eval.yaml`, runs all examples through `EvalRunner`, and prints records plus scores.

Write structured artifacts (JSON + markdown summary):

```bash
poetry run llm-eval-lab run-eval --format both
```

Reports are written to `artifacts/evals/<run_id>/report.json`.

### 5. Check for regressions against the baseline

```bash
poetry run llm-eval-lab compare-baseline artifacts/baseline.json
```

CI generates a fresh report and compares it to the committed baseline at `artifacts/baseline.json`. See [Production Eval](production-eval.md) for details.

## Configuration Overview

Evaluation behavior is controlled by `eval.yaml`:

| Setting | Purpose |
|---------|---------|
| `dataset` | Path to a JSONL file of `EvaluationExample` records |
| `tags` | Filter examples by tag (empty list = run all) |
| `metrics.local.thresholds` | Per-metric pass thresholds |
| `artifacts_dir` | Where `run-eval --format json` writes reports |
| `regression.max_mean_score_drop` | Allowed mean score drop vs baseline |

Add or edit examples in `datasets/default.jsonl`:

```json
{"question": "...", "reference_answer": "...", "reference_contexts": ["..."], "tags": ["rag"]}
```

Run only tagged examples by setting `tags: ["rag"]` in `eval.yaml`, or pass a custom config:

```bash
poetry run llm-eval-lab run-eval --config eval.yaml
```

## Enable LLM-Based Metrics

DeepEval and Ragas use an evaluator LLM (here: OpenAI) to judge quality. They remain opt-in — local metrics run in CI without an API key.

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

**Senior QA new to LLM evaluation?** Read [QA Onboarding](qa-onboarding.md) first — it maps classic QA concepts to RAG metrics and walks through every hands-on exercise in this repo.

If you see `OSError: [Errno 48] Address already in use`, another `mkdocs serve` is already running on port 8000. Either use the existing server, stop the other process, or pick a different port:

```bash
poetry run mkdocs serve -a 127.0.0.1:8001
```

## What to Read Next

1. [QA Onboarding](qa-onboarding.md) — **recommended for Senior QA new to LLM eval** (basics → advanced)
2. [Concepts](concepts.md) — understand retrieval vs generation metrics
3. [Architecture](architecture.md) — how modules connect
4. [Production Eval](production-eval.md) — config, artifacts, CI gates, plug in your app
5. [Learning Roadmap](learning-roadmap.md) — progressive skill path
6. [DeepEval](deepeval.md) and [Ragas](ragas.md) — framework-specific notes
