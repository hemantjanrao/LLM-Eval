"""Configuration loading for evaluation runs."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LocalMetricsConfig(BaseModel):
    """Settings for deterministic local metrics."""

    enabled: bool = True
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "keyword_recall": 0.55,
            "context_precision": 0.5,
            "faithfulness_heuristic": 0.45,
        }
    )


class DeepEvalMetricsConfig(BaseModel):
    """Settings for optional DeepEval LLM metrics."""

    enabled: bool = False
    threshold: float = 0.5
    model: str = "gpt-4o-mini"


class RagasMetricsConfig(BaseModel):
    """Settings for optional Ragas LLM metrics."""

    enabled: bool = False
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"


class MetricsConfig(BaseModel):
    """Metric backend toggles and thresholds."""

    local: LocalMetricsConfig = Field(default_factory=LocalMetricsConfig)
    deepeval: DeepEvalMetricsConfig = Field(default_factory=DeepEvalMetricsConfig)
    ragas: RagasMetricsConfig = Field(default_factory=RagasMetricsConfig)


class RegressionConfig(BaseModel):
    """Thresholds used when comparing runs against a baseline."""

    max_mean_score_drop: float = 0.05


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    dataset: str = "datasets/default.jsonl"
    tags: list[str] = Field(default_factory=list)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    artifacts_dir: str = "artifacts/evals"
    regression: RegressionConfig = Field(default_factory=RegressionConfig)


def repo_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """Return the default eval config file path."""

    return repo_root() / "eval.yaml"


def load_config(path: Path | None = None) -> EvalConfig:
    """Load evaluation config from YAML, falling back to defaults when missing."""

    config_path = path or default_config_path()
    if not config_path.exists():
        return EvalConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return EvalConfig()
    return EvalConfig.model_validate(raw)


def resolve_dataset_path(config: EvalConfig) -> Path:
    """Resolve a dataset path from config relative to the repository root."""

    dataset_path = Path(config.dataset)
    if dataset_path.is_absolute():
        return dataset_path
    return repo_root() / dataset_path
