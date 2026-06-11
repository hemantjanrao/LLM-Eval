"""Structured evaluation reports for CLI output and regression gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from llm_eval_lab.eval_dataset import EvaluationRunRecord
from llm_eval_lab.local_metrics import MetricResult


class RecordScore(BaseModel):
    """One metric score for one evaluated example."""

    question: str
    metric: str
    score: float
    passed: bool
    reason: str
    backend: str = "local"


class EvalRunSummary(BaseModel):
    """Aggregate statistics for an evaluation run."""

    total_examples: int
    total_scores: int
    pass_rate: float
    mean_scores: dict[str, float] = Field(default_factory=dict)
    failed_questions: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Full artifact for one evaluation run."""

    run_id: str
    timestamp: str
    git_sha: str | None = None
    config_hash: str | None = None
    records: list[EvaluationRunRecord]
    scores: list[RecordScore]
    summary: EvalRunSummary


def metric_result_to_record_score(
    record: EvaluationRunRecord,
    result: MetricResult,
    *,
    backend: str = "local",
) -> RecordScore:
    """Translate a local metric result into a framework-neutral score record."""

    return RecordScore(
        question=record.question,
        metric=result.name,
        score=result.score,
        passed=result.passed,
        reason=result.reason,
        backend=backend,
    )


def build_summary(scores: list[RecordScore], example_count: int) -> EvalRunSummary:
    """Compute aggregate pass rate and mean scores from individual metric scores."""

    if not scores:
        return EvalRunSummary(
            total_examples=example_count,
            total_scores=0,
            pass_rate=1.0,
        )

    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    failed_questions: set[str] = set()
    passed_count = 0

    for score in scores:
        metric_totals[score.metric] = metric_totals.get(score.metric, 0.0) + score.score
        metric_counts[score.metric] = metric_counts.get(score.metric, 0) + 1
        if score.passed:
            passed_count += 1
        else:
            failed_questions.add(score.question)

    mean_scores = {
        metric: round(metric_totals[metric] / metric_counts[metric], 3)
        for metric in metric_totals
    }
    return EvalRunSummary(
        total_examples=example_count,
        total_scores=len(scores),
        pass_rate=round(passed_count / len(scores), 3),
        mean_scores=mean_scores,
        failed_questions=sorted(failed_questions),
    )


def build_eval_report(
    records: list[EvaluationRunRecord],
    scores: list[RecordScore],
    *,
    config_hash: str | None = None,
    run_id: str | None = None,
) -> EvalReport:
    """Assemble a complete evaluation report from records and scores."""

    return EvalReport(
        run_id=run_id or str(uuid4()),
        timestamp=datetime.now(tz=UTC).isoformat(),
        git_sha=_best_effort_git_sha(),
        config_hash=config_hash,
        records=records,
        scores=scores,
        summary=build_summary(scores, len(records)),
    )


def config_hash_from_dict(config: dict[str, object]) -> str:
    """Return a stable short hash for a config dictionary."""

    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def write_report_json(report: EvalReport, path: Path) -> None:
    """Write an evaluation report to disk as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def write_summary_markdown(report: EvalReport, path: Path) -> None:
    """Write a human-readable markdown summary for one evaluation run."""

    lines = [
        f"# Eval Run {report.run_id}",
        "",
        f"- Timestamp: {report.timestamp}",
        f"- Git SHA: {report.git_sha or 'unknown'}",
        f"- Examples: {report.summary.total_examples}",
        f"- Pass rate: {report.summary.pass_rate:.1%}",
        "",
        "## Mean Scores",
        "",
    ]
    for metric, value in sorted(report.summary.mean_scores.items()):
        lines.append(f"- {metric}: {value:.3f}")

    if report.summary.failed_questions:
        lines.extend(["", "## Failed Questions", ""])
        lines.extend(f"- {question}" for question in report.summary.failed_questions)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_report(path: Path) -> EvalReport:
    """Load an evaluation report from a JSON file."""

    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def _best_effort_git_sha() -> str | None:
    """Return the current git commit SHA when git is available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
