# DeepEval Notes

DeepEval feels like `pytest` for LLM applications. You create test cases, choose metrics, and assert that each test case passes thresholds.

This project demonstrates:

- `LLMTestCase` for a single user interaction.
- RAG metrics such as answer relevancy, faithfulness, contextual relevancy, contextual precision, and contextual recall.
- Safety metrics such as bias and toxicity.
- `GEval` for custom rubric-style evaluation.
- A test file in `evals/deepeval/test_rag_deepeval.py`.

## Run

```bash
poetry run llm-eval-lab deepeval-smoke
poetry run deepeval test run evals/deepeval/test_rag_deepeval.py
```

Smoke and test commands load examples from `datasets/default.jsonl` via `EvalRunner`. DeepEval is not yet wired into `run-eval` — enable it explicitly in your own scripts or pytest files. See [Production Eval](production-eval.md) for the config-driven workflow.

You need an LLM provider key such as `OPENAI_API_KEY` for most LLM-as-judge metrics.

## Concept Mapping

- Answer relevancy: does the answer respond to the question?
- Faithfulness: is the answer grounded in retrieved context?
- Contextual relevancy: are retrieved chunks relevant to the question?
- Contextual precision: are the most useful chunks ranked first?
- Contextual recall: did retrieval include the facts needed for the reference answer?
- G-Eval: use natural-language criteria when no built-in metric matches your goal.

