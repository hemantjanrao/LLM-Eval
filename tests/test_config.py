"""Tests for configuration loading."""

from __future__ import annotations

from llm_eval_lab.config import load_config, repo_root, resolve_dataset_path


class TestConfig:
    """Verify YAML config parsing."""

    def test_resolve_dataset_path(self) -> None:
        """Dataset paths should resolve relative to repo root."""

        config = load_config(repo_root() / "eval.yaml")
        path = resolve_dataset_path(config)
        assert path == repo_root() / "datasets" / "default.jsonl"
        assert path.exists()

    def test_regression_defaults(self) -> None:
        """Regression config should include default threshold."""

        config = load_config(repo_root() / "eval.yaml")
        assert config.regression.max_mean_score_drop == 0.05
