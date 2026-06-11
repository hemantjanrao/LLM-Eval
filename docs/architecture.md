# Architecture

This page explains how the lab modules fit together and how data flows through evaluation.

## High-Level Flow

```mermaid
flowchart TB
    subgraph dataset [Evaluation Dataset]
        EX[EvaluationExample]
    end

    subgraph app [Toy RAG App]
        KB[ToyKnowledgeBase]
        RET[ToyRetriever]
        GEN[ToyAnswerGenerator]
        PIPE[ToyRagPipeline]
        KB --> RET
        RET --> PIPE
        GEN --> PIPE
    end

    subgraph record [Observed Result]
        REC[EvaluationRunRecord]
    end

    subgraph metrics [Scoring Layers]
        LOC[local_metrics.py]
        DE[deepeval_concepts.py]
        RA[ragas_concepts.py]
    end

    EX --> PIPE
    PIPE --> REC
    REC --> LOC
    REC --> DE
    REC --> RA
```

The same `EvaluationRunRecord` feeds all three scoring layers. That is the core design: **one neutral data model, multiple framework adapters**.

## Module Reference

### `toy_rag.py` — System Under Test

| Class | Role |
|-------|------|
| `Document` | A chunk of knowledge (id, text, topic) |
| `ToyKnowledgeBase` | In-memory document store |
| `ToyRetriever` | Keyword-overlap retrieval (top-k) |
| `ToyAnswerGenerator` | Deterministic answer synthesis |
| `ToyRagPipeline` | Wires retriever + generator |

Key methods:

- `answer(question)` — black-box use (returns only the final string)
- `run_example(example)` — evaluation use (returns full `EvaluationRunRecord`)

`build_default_pipeline()` creates the default knowledge base with four documents about DeepEval, Ragas, faithfulness, and retrieval quality.

### `eval_dataset.py` — Framework-Neutral Data

| Class | Role |
|-------|------|
| `EvaluationExample` | Input + gold answer + gold contexts + tags |
| `EvaluationRunRecord` | Example + actual answer + retrieved contexts |

| Function | Role |
|----------|------|
| `load_examples_from_file(path)` | Load examples from a JSONL file |
| `load_examples()` | Load the default dataset from `datasets/default.jsonl` |

The default dataset contains three scenarios covering DeepEval, faithfulness, and Ragas. Edit `datasets/default.jsonl` to add examples; point `eval.yaml` at a different file if needed.

### `local_metrics.py` — Deterministic Scoring

| Class | Approximates |
|-------|--------------|
| `KeywordRecallMetric` | Answer correctness vs reference |
| `ContextPrecisionMetric` | Retriever focus vs gold contexts |
| `FaithfulnessHeuristicMetric` | Answer grounding in retrieved text |

`score_record(record)` runs all three and returns `MetricResult` objects with name, score, passed, and reason.

`score_record_with_thresholds(record, thresholds)` accepts per-metric threshold overrides from `eval.yaml`.

### `deepeval_concepts.py` — DeepEval Adapter

`DeepEvalConceptFactory` translates records into DeepEval types:

| Our field | DeepEval field |
|-----------|----------------|
| `question` | `input` |
| `actual_answer` | `actual_output` |
| `reference_answer` | `expected_output` |
| `retrieved_contexts` | `retrieval_context` |
| `reference_contexts` | `context` |

Factory methods:

- `build_test_case()` → `LLMTestCase`
- `build_rag_metrics()` → answer relevancy, faithfulness, contextual metrics
- `build_safety_metrics()` → bias, toxicity
- `build_custom_correctness_metric()` → G-Eval rubric

`assert_record_with_deepeval()` runs `deepeval.assert_test()` for pytest-style pass/fail.

### `ragas_concepts.py` — Ragas Adapter

`RagasConceptFactory` translates records into Ragas types:

| Our field | Ragas field |
|-----------|-------------|
| `question` | `user_input` |
| `actual_answer` | `response` |
| `retrieved_contexts` | `retrieved_contexts` |
| `reference_answer` | `reference` |
| `reference_contexts` | `reference_contexts` |

Factory methods:

- `build_sample()` → `SingleTurnSample`
- `build_dataset()` → `EvaluationDataset`
- `build_rag_metrics()` → faithfulness, answer relevancy, context precision/recall
- `build_rubric_metric()` → `RubricsScoreWithReference`

Async helpers: `score_record_with_ragas()`, `evaluate_records_with_ragas()`.

### `compat.py` — Dependency Shims

`install_ragas_vertexai_import_shim()` prevents Ragas 0.4 from failing on an optional VertexAI import when you only use OpenAI.

### `cli.py` — Learning Interface

| Command | What it does |
|---------|--------------|
| `ask` | Query the toy RAG app |
| `inspect-dataset` | Show all evaluation records |
| `score-local` | Run deterministic metrics |
| `run-eval` | Full suite with optional JSON artifacts |
| `compare-baseline` | Regression gate vs committed baseline |
| `deepeval-smoke` | One DeepEval assertion (needs API key) |
| `ragas-smoke` | One Ragas metric (needs API key) |

`ConsoleReporter` and `JsonReporter` separate presentation from scoring.

### `runner.py` — Eval Orchestration

| Component | Role |
|-----------|------|
| `EvaluableApp` | Protocol for pluggable apps under test |
| `EvalRunner` | Loads config/dataset, runs examples, scores with enabled backends |
| `filter_examples_by_tags()` | Tag-based example filtering |

### `config.py` — YAML Configuration

Loads `eval.yaml` for dataset path, metric thresholds, artifacts directory, and regression settings.

### `reporting.py` — Structured Reports

| Model | Role |
|-------|------|
| `RecordScore` | One metric score for one example |
| `EvalRunSummary` | Pass rate and mean scores |
| `EvalReport` | Full run artifact (JSON + markdown summary) |

### `regression.py` — Baseline Gates

`compare_reports()` detects mean score drops and pass-to-fail flips against `artifacts/baseline.json`.

## Production Flow

```mermaid
flowchart TB
    CFG[eval.yaml]
    DS[datasets/default.jsonl]
    APP[EvaluableApp]
    RUN[EvalRunner]
    RPT[EvalReport JSON]
    GATE[compare_reports]

    CFG --> RUN
    DS --> RUN
    APP --> RUN
    RUN --> RPT --> GATE
```

## Test Layout

```
tests/                          # API-free unit tests (run in CI)
  test_runner.py                # EvalRunner, golden mean scores, tag filtering
  test_regression.py            # Baseline comparison
  test_config.py                # eval.yaml loading
  test_local_metrics.py         # Per-example golden scores
evals/deepeval/                 # DeepEval test file (needs API key)
examples/                       # Standalone demo scripts
.github/workflows/ci.yml        # ruff, mypy, pytest, regression gate
artifacts/baseline.json          # Committed known-good EvalReport
```

## Extension Points

To evaluate your own RAG app:

1. Implement `EvaluableApp` (see `docs/production-eval.md`) and pass it to `EvalRunner().run(app=...)`.
2. Add examples to `datasets/default.jsonl` or point `eval.yaml` at your own JSONL file.
3. Reuse `score_record()`, `assert_record_with_deepeval()`, or `evaluate_records_with_ragas()`.
4. Add local metrics in `local_metrics.py` for fast regression checks.
5. Run `poetry run llm-eval-lab run-eval --format both` and gate with `compare-baseline`.
