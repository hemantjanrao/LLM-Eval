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

Read `src/llm_eval_lab/eval_dataset.py`.

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

**Checkpoint:** You can describe what an `EvaluationRunRecord` adds on top of an `EvaluationExample`.

## Level 3: Local Metrics (No API Key)

**Goal:** Learn metric intuition with deterministic scores.

Run:
```bash
poetry run llm-eval-lab score-local
```

Read `src/llm_eval_lab/local_metrics.py`:

| Metric | Teaches |
|--------|---------|
| `keyword_recall` | Reference-based answer correctness |
| `context_precision` | Retriever focus on gold evidence |
| `faithfulness_heuristic` | Answer grounding in retrieved text |

Notice the faithfulness example: answer quality passes but `context_precision` fails. That means retrieval ranking needs work, not generation.

Run `poetry run pytest` — all tests pass without an API key.

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

## Level 6: CI and Production Patterns

**Goal:** Make evals part of your workflow.

| When | What to run |
|------|-------------|
| Every commit | `pytest` (local metrics, toy RAG) |
| PR touching prompts/retrieval | DeepEval or Ragas with API key |
| Scheduled / pre-release | Full dataset evaluation with LLM metrics |

Patterns from this repo:

- Skip LLM tests when `OPENAI_API_KEY` is missing (`evals/deepeval/test_rag_deepeval.py`)
- Keep framework adapters thin — data model stays in `eval_dataset.py`
- Use smoke commands to verify wiring before full eval runs

**Checkpoint:** You have a plan for which metrics run where in your pipeline.

## Level 7: Evaluate Your Own App

Replace the toy pipeline:

1. Implement your RAG app's `run_example()` to return `EvaluationRunRecord`
2. Add real examples to `load_examples()` or load from a file
3. Start with local metrics, then add DeepEval/Ragas
4. Tune thresholds based on your domain

See [Architecture](architecture.md) for extension points.
