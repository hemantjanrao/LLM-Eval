# Learning Roadmap

## Level 1: Evaluation Data

Start with `EvaluationExample` in `src/llm_eval_lab/eval_dataset.py`.

Each example contains:

- `question`: the user input.
- `reference_answer`: the ideal answer.
- `reference_contexts`: evidence that should support the answer.
- `tags`: labels for filtering and reporting.

## Level 2: Local Metrics

Run:

```bash
poetry run llm-eval-lab score-local
```

The local metrics are deliberately simple:

- Answer keyword recall approximates answer correctness.
- Context precision approximates retriever ranking quality.
- Faithfulness heuristic checks whether answer terms are grounded in retrieved context.

## Level 3: Framework Metrics

DeepEval and Ragas add stronger LLM-based judgments. They can evaluate criteria that simple string matching misses, such as nuanced hallucination, answer relevancy, and rubric-based quality.

## Level 4: CI

Use deterministic tests for every commit. Use LLM-based evals on scheduled runs, pull requests touching prompts/retrieval, or before release.

