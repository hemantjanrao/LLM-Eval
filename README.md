# LLM Eval Lab

A hands-on learning project for DeepEval and Ragas.

This repo teaches evaluation concepts with a tiny Retrieval-Augmented Generation (RAG) app, deterministic local checks, and optional LLM-as-judge examples using DeepEval and Ragas.

## What You Will Learn

- Build an evaluation dataset with inputs, expected answers, retrieved contexts, and references.
- Separate retriever quality from generator quality.
- Use deterministic metrics before moving to LLM-as-judge metrics.
- Run DeepEval-style tests for answer relevancy, faithfulness, contextual precision/recall, toxicity, bias, and custom G-Eval rubrics.
- Run Ragas-style single-turn metrics and dataset evaluation.
- Add evals to normal `pytest` and CI-friendly workflows.

## Setup

Poetry is used for dependency management.

```bash
poetry env use 3.13  # optional, but recommended when Python 3.13 is installed
poetry install --with dev,docs
poetry run pytest
```

The deterministic tests do not require an API key.

For DeepEval and Ragas LLM metrics:

```bash
cp .env.example .env
# edit .env and add OPENAI_API_KEY
poetry run llm-eval-lab deepeval-smoke
poetry run llm-eval-lab ragas-smoke
```

## Useful Commands

```bash
poetry run llm-eval-lab ask "What is DeepEval useful for?"
poetry run llm-eval-lab inspect-dataset
poetry run llm-eval-lab score-local
poetry run llm-eval-lab run-eval --format both
poetry run llm-eval-lab compare-baseline artifacts/ci-report.json
poetry run pytest
poetry run ruff check .
poetry run mypy
poetry run mkdocs serve
```

## Production Eval Workflow

- [`eval.yaml`](eval.yaml): config for dataset path, metric thresholds, and artifacts
- [`datasets/default.jsonl`](datasets/default.jsonl): versioned evaluation examples
- [`artifacts/baseline.json`](artifacts/baseline.json): committed baseline for regression gates
- [`docs/production-eval.md`](docs/production-eval.md): plug in your app, run CI gates

## Project Map

- `src/llm_eval_lab/toy_rag.py`: a small documented RAG pipeline.
- `src/llm_eval_lab/eval_dataset.py`: reusable evaluation examples (JSONL-backed).
- `src/llm_eval_lab/runner.py`: eval orchestration and pluggable app interface.
- `src/llm_eval_lab/config.py`: YAML config loading.
- `src/llm_eval_lab/reporting.py`: structured eval reports and artifacts.
- `src/llm_eval_lab/regression.py`: baseline comparison and regression gates.
- `src/llm_eval_lab/local_metrics.py`: deterministic metrics that explain the intuition behind RAG evaluation.
- `src/llm_eval_lab/deepeval_concepts.py`: DeepEval test cases and metrics.
- `src/llm_eval_lab/ragas_concepts.py`: Ragas samples, datasets, and metrics.
- `evals/deepeval/test_rag_deepeval.py`: DeepEval test file you can run with `deepeval test run`.
- `docs/`: concept notes and learning path.

## Documentation

Full docs live in `docs/`. Serve them locally with:

```bash
poetry run mkdocs serve
```

| Doc | Topic |
|-----|-------|
| `docs/qa-onboarding.md` | **Start here (QA)** — basics to advanced LLM eval onboarding |
| `docs/getting-started.md` | Install and first commands |
| `docs/concepts.md` | RAG evaluation metrics explained |
| `docs/architecture.md` | How modules connect |
| `docs/production-eval.md` | Production workflow, CI gates, plug-in your app |
| `docs/learning-roadmap.md` | Step-by-step learning path |
| `docs/deepeval.md` | DeepEval framework notes |
| `docs/ragas.md` | Ragas framework notes |

## Recommended Learning Path

**Senior QA new to LLM eval:** read [`docs/qa-onboarding.md`](docs/qa-onboarding.md) first (basics → metrics → CI → production).

1. Run `poetry run llm-eval-lab ask "What is Ragas?"`.
2. Read `src/llm_eval_lab/toy_rag.py` to understand the app being evaluated.
3. Run `poetry run llm-eval-lab run-eval` and inspect metric explanations.
4. Read `docs/concepts.md` and `docs/architecture.md`.
5. Read `docs/production-eval.md` — config, artifacts, CI gates, plug in your app.
6. Read `docs/deepeval.md`, then run the DeepEval smoke command with an API key.
7. Read `docs/ragas.md`, then run the Ragas smoke command with an API key.
