"""Tests for evaluation orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_lab.config import EvalConfig, load_config, repo_root
from llm_eval_lab.eval_dataset import load_examples, load_examples_from_file
from llm_eval_lab.runner import EvalRunner, filter_examples_by_tags, run_default_eval


class TestDatasetLoading:
    """Verify external dataset loading."""

    def test_load_examples_from_default_jsonl(self) -> None:
        """Default dataset should contain three examples."""

        examples = load_examples()
        assert len(examples) == 3
        assert all(example.question for example in examples)

    def test_load_examples_from_file(self) -> None:
        """JSONL loader should parse the default dataset file."""

        path = repo_root() / "datasets" / "default.jsonl"
        examples = load_examples_from_file(path)
        assert len(examples) == 3


class TestConfig:
    """Verify eval.yaml loading."""

    def test_load_default_config(self) -> None:
        """Default config should enable local metrics."""

        config = load_config(repo_root() / "eval.yaml")
        assert config.metrics.local.enabled is True
        assert config.metrics.deepeval.enabled is False


class TestEvalRunner:
    """Verify runner behavior and golden local metric scores."""

    def test_run_default_eval_produces_three_records(self) -> None:
        """Default run should evaluate all three examples."""

        report = run_default_eval()
        assert len(report.records) == 3
        assert len(report.scores) == 9

    def test_golden_mean_scores(self) -> None:
        """Mean scores should remain stable for the toy RAG pipeline."""

        report = run_default_eval()
        assert report.summary.mean_scores == {
            "keyword_recall": 0.698,
            "context_precision": 0.475,
            "faithfulness_heuristic": 0.776,
        }
        assert report.summary.pass_rate == pytest.approx(0.889)

    def test_filter_examples_by_tags(self) -> None:
        """Tag filtering should return only matching examples."""

        examples = load_examples()
        filtered = filter_examples_by_tags(examples, ["rag"])
        assert len(filtered) == 1
        assert filtered[0].tags == ["rag", "metric"]

    def test_runner_respects_tag_filter_in_config(self) -> None:
        """Runner should honor tag filters from config."""

        config = EvalConfig(tags=["deepeval"])
        report = EvalRunner(config).run()
        assert len(report.records) == 1
        assert "deepeval" in report.records[0].example.tags

    def test_empty_dataset_path_raises(self, tmp_path: Path) -> None:
        """Empty JSONL files should raise a clear error."""

        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No evaluation examples"):
            load_examples_from_file(empty_file)
