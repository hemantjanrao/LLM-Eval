# LLM Eval Lab

A hands-on learning project for **DeepEval** and **Ragas** — two popular frameworks for evaluating LLM and RAG applications.

## Why This Project Exists

Most teams evaluate LLM apps with informal "vibe checks." That does not scale. This lab teaches you to:

- Build structured evaluation datasets
- Separate retriever quality from generator quality
- Use fast deterministic checks before expensive LLM-as-judge metrics
- Integrate evals into `pytest` and CI workflows

The code starts with a **tiny, deterministic RAG app** so every failure is understandable. Once local metrics make sense, the DeepEval and Ragas examples show how the same ideas become semantic, LLM-powered judgments.

## Quick Start

```bash
poetry install --with dev,docs
poetry run pytest                              # no API key needed
poetry run llm-eval-lab ask "What is Ragas?"
poetry run llm-eval-lab score-local
```

For LLM metrics, set `OPENAI_API_KEY` in `.env` and run:

```bash
poetry run llm-eval-lab deepeval-smoke
poetry run llm-eval-lab ragas-smoke
```

See [Getting Started](getting-started.md) for the full walkthrough.

## Documentation Map

| Page | What you will learn |
|------|---------------------|
| [Getting Started](getting-started.md) | Install, first commands, API setup |
| [Concepts](concepts.md) | RAG metrics explained with examples |
| [Architecture](architecture.md) | How modules connect and data flows |
| [Learning Roadmap](learning-roadmap.md) | Progressive skill path |
| [DeepEval](deepeval.md) | pytest-style LLM testing |
| [Ragas](ragas.md) | Dataset-driven evaluation loops |

## Core Evaluation Questions

LLM evaluation is not one score. It is a set of questions:

- Did retrieval find the right evidence?
- Did generation stay faithful to that evidence?
- Did the answer address the user?
- Did the answer match the reference answer?
- Did the system avoid unsafe, biased, or toxic behavior?
- Did the workflow succeed end to end?

This lab gives you tools to answer each question systematically.
