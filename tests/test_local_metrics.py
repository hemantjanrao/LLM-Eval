"""Tests for deterministic learning metrics."""

from llm_eval_lab.eval_dataset import load_examples
from llm_eval_lab.local_metrics import MetricResult, score_record
from llm_eval_lab.toy_rag import build_default_pipeline


class TestLocalMetrics:
    """Check that local metric output is predictable and learner-friendly."""

    def test_score_record_returns_named_results(self) -> None:
        """Each evaluation record should produce multiple explained metric scores."""

        pipeline = build_default_pipeline()
        record = pipeline.run_example(load_examples()[0])
        results = score_record(record)
        assert len(results) == 3
        assert all(isinstance(result, MetricResult) for result in results)
        assert all(0.0 <= result.score <= 1.0 for result in results)
        assert all(result.reason for result in results)

