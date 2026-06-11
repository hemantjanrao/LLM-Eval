# Learning Roadmap

Follow these levels in order. Each level builds on the previous one.

## Level 1: Understand the App Being Evaluated

**Goal:** Know what you are testing before you score it.

1. Run:
   ```bash
   poetry run llm-eval-lab ask "What is DeepEval useful for?"
   ```
2. Read `src/llm_eval_lab/toy_rag.py`:
   - `ToyRetriever` ranks documents by keyword overlap
   - `ToyAnswerGenerator` returns canned answers for known topics
   - `ToyRagPipeline.run_example()` preserves retrieved contexts for metrics

**Checkpoint:** You can explain the difference between `answer()` (black box) and `run_example()` (evaluation mode).

## Level 2: Evaluation Data

**Goal:** Understand the gold standard.

Read `src/llm_eval_lab/eval_dataset.py` and `datasets/default.jsonl`.

Each `EvaluationExample` contains:

| Field | Purpose |
|-------|---------|
| `question` | User input sent to the app |
| `reference_answer` | Ideal answer (gold) |
| `reference_contexts` | Evidence that should support the answer |
| `tags` | Labels for filtering reports |

Run:
```bash
poetry run llm-eval-lab inspect-dataset
```

**Checkpoint:** You can describe what an `EvaluationRunRecord` adds on top of an `EvaluationExample`, and where examples are loaded from (JSONL, not hardcoded Python).

## Level 3: Local Metrics (No API Key)

**Goal:** Learn metric intuition with deterministic scores.

Run:
```bash
poetry run llm-eval-lab score-local
# or the full suite entry point:
poetry run llm-eval-lab run-eval
```

Read `src/llm_eval_lab/local_metrics.py`:

| Metric | Teaches |
|--------|---------|
| `keyword_recall` | Reference-based answer correctness |
| `context_precision` | Retriever focus on gold evidence |
| `faithfulness_heuristic` | Answer grounding in retrieved text |

Notice the faithfulness example: answer quality passes but `context_precision` fails. That means retrieval ranking needs work, not generation.

Run `poetry run pytest` — all 19 tests pass without an API key, including golden score assertions.

**Checkpoint:** You can read a metric table and diagnose whether retrieval or generation is the problem.

## Level 4: DeepEval (LLM-as-Judge)

**Goal:** Use pytest-style assertions with semantic metrics.

1. Read [DeepEval notes](deepeval.md)
2. Read `src/llm_eval_lab/deepeval_concepts.py`
3. Set `OPENAI_API_KEY` in `.env`
4. Run:
   ```bash
   poetry run llm-eval-lab deepeval-smoke
   poetry run deepeval test run evals/deepeval/test_rag_deepeval.py
   ```

Key concepts:

- `LLMTestCase` — one interaction with input, output, contexts
- RAG metrics — relevancy, faithfulness, contextual precision/recall
- Safety metrics — bias, toxicity
- G-Eval — custom rubrics in natural language

**Checkpoint:** You can map our `EvaluationRunRecord` fields to DeepEval's `LLMTestCase` fields.

## Level 5: Ragas (Dataset Evaluation)

**Goal:** Evaluate collections of samples systematically.

1. Read [Ragas notes](ragas.md)
2. Read `src/llm_eval_lab/ragas_concepts.py`
3. Run:
   ```bash
   poetry run llm-eval-lab ragas-smoke
   python examples/run_ragas_one_case.py
   ```

Key concepts:

- `SingleTurnSample` — one evaluated interaction
- `EvaluationDataset` — batch of samples
- Async scoring with `single_turn_ascore()`
- `evaluate()` for full dataset runs

**Checkpoint:** You can explain when to use DeepEval (test assertions) vs Ragas (dataset loops).

## Level 6: Production Eval and CI

**Goal:** Make evals part of your workflow with config, artifacts, and regression gates.

Read [Production Eval](production-eval.md) and `eval.yaml`.

Run:
```bash
poetry run llm-eval-lab run-eval --format both
poetry run llm-eval-lab compare-baseline artifacts/baseline.json
```

| When | What to run | API key? |
|------|-------------|----------|
| Every commit / PR | `pytest` + regression gate (CI) | No |
| After app changes | `run-eval --format both` | No |
| PR touching prompts/retrieval | DeepEval or Ragas locally | Yes |
| Scheduled / pre-release | Full LLM dataset evaluation | Yes |

Key modules:

| Module | Role |
|--------|------|
| `runner.py` | `EvalRunner`, `EvaluableApp` protocol |
| `config.py` | Load `eval.yaml` |
| `reporting.py` | `EvalReport` artifacts |
| `regression.py` | Baseline comparison |

Patterns from this repo:

- Local metrics and golden tests run in CI without secrets
- LLM tests skip when `OPENAI_API_KEY` is missing (`evals/deepeval/test_rag_deepeval.py`)
- Committed baseline at `artifacts/baseline.json` gates score regressions
- Framework adapters stay thin — data model stays in `eval_dataset.py`

**Checkpoint:** You can explain what `run-eval`, `compare-baseline`, and the CI workflow do.

## Level 7: Evaluate Your Own App

Replace the toy pipeline:

1. Implement `EvaluableApp.run_example()` to return `EvaluationRunRecord`
2. Add examples to `datasets/default.jsonl` or point `eval.yaml` at your own file
3. Start with `run-eval` (local metrics), then add DeepEval/Ragas
4. Tune thresholds in `eval.yaml` based on your domain
5. Update `artifacts/baseline.json` after intentional improvements

See [Production Eval](production-eval.md) and [Architecture](architecture.md) for extension points.
