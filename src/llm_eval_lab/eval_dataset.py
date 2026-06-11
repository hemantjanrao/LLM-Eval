"""Small evaluation dataset used across local, DeepEval, and Ragas examples."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    """A single human-readable scenario for evaluating a RAG application.

    The class is intentionally framework-neutral. DeepEval calls this kind of object
    a test case or golden, while Ragas calls the evaluated object a sample. Keeping
    our own small schema makes the learning flow clearer because you can translate
    the same example into each framework.
    """

    question: str = Field(description="The user input sent to the application.")
    reference_answer: str = Field(description="The ideal answer we expect the app to produce.")
    reference_contexts: list[str] = Field(
        description="Gold evidence that should support the reference answer."
    )
    tags: list[str] = Field(default_factory=list, description="Labels for filtering reports.")


class EvaluationRunRecord(BaseModel):
    """The fully observed result of running one example through the toy RAG app.

    Evals become useful when you store both expected and observed behavior. This
    class joins the dataset example with the actual answer and retrieved contexts
    produced by the current system version.
    """

    example: EvaluationExample
    actual_answer: str
    retrieved_contexts: list[str]

    @property
    def question(self) -> str:
        """Return the original user question for reporting convenience."""

        return self.example.question


def load_examples() -> list[EvaluationExample]:
    """Return a compact dataset that covers retrieval, generation, and safety basics."""

    return [
        EvaluationExample(
            question="What is DeepEval useful for?",
            reference_answer=(
                "DeepEval is useful for testing LLM applications with metrics such as "
                "answer relevancy, faithfulness, contextual metrics, and custom G-Eval rubrics."
            ),
            reference_contexts=[
                "DeepEval is an LLM evaluation framework that works like pytest for LLM apps.",
                (
                    "DeepEval includes answer relevancy, faithfulness, contextual RAG "
                    "metrics, and G-Eval."
                ),
            ],
            tags=["deepeval", "framework"],
        ),
        EvaluationExample(
            question="What does faithfulness measure in RAG evaluation?",
            reference_answer=(
                "Faithfulness measures whether the generated answer is supported by the "
                "retrieved context instead of inventing unsupported claims."
            ),
            reference_contexts=[
                (
                    "Faithfulness evaluates whether a response is factually grounded in "
                    "retrieved context."
                ),
                (
                    "A faithful RAG answer should only make claims supported by the "
                    "retrieved evidence."
                ),
            ],
            tags=["rag", "metric"],
        ),
        EvaluationExample(
            question="How is Ragas different from a vibe check?",
            reference_answer=(
                "Ragas helps replace subjective vibe checks with repeatable evaluation datasets, "
                "metrics, and feedback loops for LLM applications."
            ),
            reference_contexts=[
                "Ragas helps teams move from vibe checks to systematic evaluation loops.",
                "Ragas provides metrics and test generation for LLM and RAG applications.",
            ],
            tags=["ragas", "framework"],
        ),
    ]
