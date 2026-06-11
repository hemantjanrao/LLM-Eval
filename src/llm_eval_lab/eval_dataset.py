"""Small evaluation dataset used across local, DeepEval, and Ragas examples."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from llm_eval_lab.config import repo_root


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


DEFAULT_DATASET_PATH = repo_root() / "datasets" / "default.jsonl"


def load_examples_from_file(path: Path) -> list[EvaluationExample]:
    """Load evaluation examples from a JSONL file."""

    examples: list[EvaluationExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        try:
            examples.append(EvaluationExample.model_validate(payload))
        except Exception as error:
            raise ValueError(f"Invalid example on line {line_number} of {path}") from error
    if not examples:
        raise ValueError(f"No evaluation examples found in {path}")
    return examples


def load_examples(path: Path | None = None) -> list[EvaluationExample]:
    """Return the default evaluation dataset from JSONL."""

    dataset_path = path or DEFAULT_DATASET_PATH
    return load_examples_from_file(dataset_path)
