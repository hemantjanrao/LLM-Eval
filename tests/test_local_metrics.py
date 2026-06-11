"""Tests for deterministic learning metrics."""

from llm_eval_lab.eval_dataset import load_examples
from llm_eval_lab.local_metrics import MetricResult, score_record
from llm_eval_lab.toy_rag import build_default_pipeline

GOLDEN_SCORES = {
    "What is DeepEval useful for?": {
        "keyword_recall": 0.571,
        "context_precision": 0.502,
        "faithfulness_heuristic": 0.714,
    },
    "What does faithfulness measure in RAG evaluation?": {
        "keyword_recall": 0.667,
        "context_precision": 0.416,
        "faithfulness_heuristic": 0.9,
    },
    "How is Ragas different from a vibe check?": {
        "keyword_recall": 0.857,
        "context_precision": 0.508,
        "faithfulness_heuristic": 0.714,
    },
}


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

    def test_golden_scores_per_example(self) -> None:
        """Local metric scores should match known-good values for the toy RAG."""

        pipeline = build_default_pipeline()
        for example in load_examples():
            record = pipeline.run_example(example)
            results = {result.name: result.score for result in score_record(record)}
            assert results == GOLDEN_SCORES[example.question]

