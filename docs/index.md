# LLM Eval Lab

This project is a guided lab for learning LLM evaluation.

The code intentionally starts with a tiny RAG system so every evaluation failure is understandable. Once the local metrics make sense, the DeepEval and Ragas examples show how the same ideas become LLM-as-judge metrics.

## Core Idea

LLM evaluation is not one score. It is a set of questions:

- Did retrieval find the right evidence?
- Did generation stay faithful to that evidence?
- Did the answer address the user?
- Did the answer match the reference answer?
- Did the system avoid unsafe, biased, or toxic behavior?
- Did the workflow succeed end to end?

