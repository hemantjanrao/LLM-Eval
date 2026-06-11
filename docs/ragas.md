# Ragas Notes

Ragas focuses on systematic evaluation loops for LLM and RAG applications. It represents interactions as samples and evaluates them with metrics.

This project demonstrates:

- `SingleTurnSample` examples.
- `EvaluationDataset` construction.
- RAG metrics such as faithfulness, answer relevancy, context precision, and context recall.
- General-purpose rubric metrics through `RubricsScoreWithReference`.

## Run

```bash
poetry run llm-eval-lab ragas-smoke
```

You need an LLM provider key such as `OPENAI_API_KEY` for metrics that call an evaluator model.

## Concept Mapping

- Single-turn sample: one question, answer, retrieved context, and reference.
- Evaluation dataset: a collection of comparable samples.
- Faithfulness: whether response claims are supported by retrieved context.
- Answer relevancy: whether the response addresses the user input.
- Context metrics: whether retrieval found and ranked useful evidence.
- Rubric scoring: whether a response satisfies a custom quality scale.

## Compatibility Note

Ragas 0.4.3 currently imports an optional VertexAI integration from an older LangChain path.
The project adapter installs a narrow shim before importing Ragas so OpenAI-based examples
continue to run while that upstream issue is active.
