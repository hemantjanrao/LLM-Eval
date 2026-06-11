"""DeepEval tests for the toy RAG pipeline.

Run this file with:

    poetry run deepeval test run evals/deepeval/test_rag_deepeval.py

Most metrics call an evaluator LLM, so set `OPENAI_API_KEY` before running.
"""

from __future__ import annotations

import os

import pytest

from llm_eval_lab.deepeval_concepts import assert_record_with_deepeval
from llm_eval_lab.eval_dataset import load_examples
from llm_eval_lab.toy_rag import build_default_pipeline

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="DeepEval LLM metrics require OPENAI_API_KEY.",
)


class TestDeepEvalRagConcepts:
    """Group DeepEval examples so pytest output reads like a learning checklist."""

    @pytest.mark.parametrize("example", load_examples())
    def test_rag_answer_quality(self, example) -> None:  # type: ignore[no-untyped-def]
        """Assert that each toy RAG answer passes the configured DeepEval metrics."""

        pipeline = build_default_pipeline()
        record = pipeline.run_example(example)
        assert_record_with_deepeval(record, threshold=0.5)

