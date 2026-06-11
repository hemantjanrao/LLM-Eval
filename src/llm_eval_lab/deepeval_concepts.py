"""DeepEval adapters for the toy RAG examples."""

from __future__ import annotations

from typing import Any, cast

from llm_eval_lab.eval_dataset import EvaluationRunRecord


class DeepEvalConceptFactory:
    """Build DeepEval-native objects from framework-neutral evaluation records.

    Imports happen inside methods so the rest of the project can run even before
    optional DeepEval dependencies are installed or configured. This also makes the
    translation layer explicit: our record becomes an `LLMTestCase`, and selected
    metrics become the assertions.
    """

    def build_test_case(self, record: EvaluationRunRecord) -> Any:
        """Convert one toy RAG run into a DeepEval `LLMTestCase`."""

        from deepeval.test_case import LLMTestCase

        return LLMTestCase(
            input=record.example.question,
            actual_output=record.actual_answer,
            expected_output=record.example.reference_answer,
            retrieval_context=cast(Any, record.retrieved_contexts),
            context=record.example.reference_contexts,
        )

    def build_rag_metrics(self, threshold: float = 0.5) -> list[Any]:
        """Return core DeepEval RAG metrics for answer and retrieval quality."""

        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        )

        return [
            AnswerRelevancyMetric(threshold=threshold),
            FaithfulnessMetric(threshold=threshold),
            ContextualRelevancyMetric(threshold=threshold),
            ContextualPrecisionMetric(threshold=threshold),
            ContextualRecallMetric(threshold=threshold),
        ]

    def build_safety_metrics(self, threshold: float = 0.5) -> list[Any]:
        """Return DeepEval safety metrics that inspect problematic language."""

        from deepeval.metrics import BiasMetric, ToxicityMetric

        return [
            BiasMetric(threshold=threshold),
            ToxicityMetric(threshold=threshold),
        ]

    def build_custom_correctness_metric(self, threshold: float = 0.5) -> Any:
        """Return a G-Eval rubric for reference-based answer correctness."""

        from deepeval.metrics import GEval
        from deepeval.test_case import SingleTurnParams

        return GEval(
            name="Reference Correctness",
            criteria=(
                "Judge whether the actual output correctly answers the input using "
                "the expected output as the reference. Penalize unsupported extra claims."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=threshold,
        )


def assert_record_with_deepeval(record: EvaluationRunRecord, threshold: float = 0.5) -> None:
    """Run a record through DeepEval assertions.

    This function is intentionally thin because DeepEval owns the test reporting.
    Use it when you want `deepeval test run` or pytest-style failures.
    """

    import deepeval

    factory = DeepEvalConceptFactory()
    test_case = factory.build_test_case(record)
    metrics = factory.build_rag_metrics(threshold=threshold)
    metrics.append(factory.build_custom_correctness_metric(threshold=threshold))
    deepeval.assert_test(test_case, metrics)  # type: ignore[attr-defined]
