# RAG Evaluation Concepts

LLM evaluation is not one number. It is a set of questions about different parts of your system. This lab teaches you to ask those questions systematically.

## The RAG Pipeline Under Test

A Retrieval-Augmented Generation (RAG) app has two main stages:

```mermaid
flowchart LR
    Q[User Question] --> R[Retriever]
    R --> C[Retrieved Contexts]
    C --> G[Generator / LLM]
    G --> A[Final Answer]
```

**Retriever quality** and **generator quality** are separate concerns. A perfect generator cannot fix bad retrieval, and perfect retrieval does not help if the generator hallucinates.

## The Evaluation Record

Every metric in this lab inspects the same object: an `EvaluationRunRecord`.

| Field | Meaning |
|-------|---------|
| `question` | What the user asked |
| `reference_answer` | The ideal answer (gold standard) |
| `reference_contexts` | Evidence that should support the answer |
| `actual_answer` | What your app produced |
| `retrieved_contexts` | What your retriever returned |

You need **both expected and observed** behavior to evaluate. The dataset (`EvaluationExample`) holds expectations; running the app produces observations (`EvaluationRunRecord`).

## Metric Families

### 1. Answer quality (generation)

Does the final answer satisfy the user?

| Metric | Question it asks |
|--------|------------------|
| **Keyword recall** (local) | Do important terms from the reference answer appear in the actual answer? |
| **Answer relevancy** (DeepEval / Ragas) | Does the answer address the user's question? |
| **G-Eval / Rubric** (DeepEval / Ragas) | Does the answer meet a custom natural-language rubric? |

### 2. Faithfulness / grounding (generation + retrieval)

Is the answer supported by retrieved evidence?

| Metric | Question it asks |
|--------|------------------|
| **Faithfulness heuristic** (local) | Do answer terms overlap with retrieved context? |
| **Faithfulness** (DeepEval / Ragas) | Are claims in the answer grounded in retrieved context? |

A high answer-quality score with low faithfulness often means **hallucination**.

### 3. Retrieval quality

Did the retriever find the right evidence?

| Metric | Question it asks |
|--------|------------------|
| **Context precision** (local / DeepEval / Ragas) | Of what was retrieved, how much was useful? |
| **Context recall** (DeepEval / Ragas) | Did retrieval include all facts needed for the reference answer? |
| **Contextual relevancy** (DeepEval) | Are retrieved chunks relevant to the question? |

**Precision** = "of what I retrieved, how much was good?"  
**Recall** = "of what I needed, how much did I retrieve?"

### 4. Safety

Does the output avoid harmful content?

| Metric | Question it asks |
|--------|------------------|
| **Toxicity** (DeepEval) | Does the output contain toxic language? |
| **Bias** (DeepEval) | Does the output show unfair bias? |

## Local vs LLM-as-Judge

| Approach | Pros | Cons |
|----------|------|------|
| **Local (deterministic)** | Free, fast, CI-friendly, easy to debug | Misses semantics, synonyms, nuance |
| **LLM-as-judge** | Handles nuance, hallucination, relevancy | Costs money, slower, needs API key |

**Recommended workflow:**

1. Start with local metrics to understand failures (`run-eval` or `score-local`).
2. Add LLM-as-judge metrics for semantic quality (DeepEval / Ragas smoke commands).
3. Run deterministic tests and regression gates on every commit; run LLM evals on schedule or before release.

See [Production Eval](production-eval.md) for the full CI and artifact workflow.

## Structured Reports

Beyond per-metric `MetricResult` objects, the runner produces an `EvalReport` with:

| Field | Meaning |
|-------|---------|
| `scores` | All `RecordScore` entries (question, metric, score, passed, reason) |
| `summary.pass_rate` | Fraction of metric checks that passed |
| `summary.mean_scores` | Average score per metric across examples |
| `summary.failed_questions` | Questions with at least one failing metric |

Use `poetry run llm-eval-lab run-eval --format both` to write JSON artifacts, then `compare-baseline` to gate regressions against `artifacts/baseline.json`.

## Reading Scores

Every metric returns:

- **name** — which metric ran
- **score** — typically 0.0 to 1.0
- **passed** — whether score met the threshold
- **reason** — human-readable explanation

When a metric fails, ask:

1. Is retrieval wrong? → fix retriever, chunking, or embeddings.
2. Is generation wrong but retrieval OK? → fix prompts, model, or grounding instructions.
3. Is the reference answer wrong? → fix your dataset.

## Example Failure from This Lab

Running `score-local` on the faithfulness question shows `context_precision` failing (score ~0.42). The answer is good and faithfulness is high, but the retriever ranked a less-relevant chunk alongside the right one. That is a **retrieval ranking** problem — exactly what context precision is designed to catch.
