"""Regression comparison between evaluation reports."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_eval_lab.config import RegressionConfig
from llm_eval_lab.reporting import EvalReport


class RegressionIssue(BaseModel):
    """One detected regression between current and baseline reports."""

    kind: str
    message: str


class RegressionResult(BaseModel):
    """Outcome of comparing a current report against a baseline."""

    passed: bool
    issues: list[RegressionIssue] = Field(default_factory=list)


def compare_reports(
    current: EvalReport,
    baseline: EvalReport,
    *,
    regression: RegressionConfig | None = None,
) -> RegressionResult:
    """Detect metric regressions and pass-to-fail flips against a baseline report."""

    settings = regression or RegressionConfig()
    issues: list[RegressionIssue] = []

    for metric, baseline_mean in baseline.summary.mean_scores.items():
        current_mean = current.summary.mean_scores.get(metric)
        if current_mean is None:
            issues.append(
                RegressionIssue(
                    kind="missing_metric",
                    message=f"Metric '{metric}' present in baseline but missing in current run.",
                )
            )
            continue
        drop = baseline_mean - current_mean
        if drop > settings.max_mean_score_drop:
            issues.append(
                RegressionIssue(
                    kind="mean_score_drop",
                    message=(
                        f"Metric '{metric}' mean dropped by {drop:.3f} "
                        f"(baseline={baseline_mean:.3f}, current={current_mean:.3f})."
                    ),
                )
            )

    baseline_passes = {
        (score.question, score.metric): score.passed
        for score in baseline.scores
        if score.backend == "local"
    }
    for score in current.scores:
        if score.backend != "local":
            continue
        key = (score.question, score.metric)
        baseline_passed = baseline_passes.get(key)
        if baseline_passed is True and not score.passed:
            issues.append(
                RegressionIssue(
                    kind="pass_to_fail",
                    message=(
                        f"Example '{score.question}' regressed on metric "
                        f"'{score.metric}' (passed in baseline, failed now)."
                    ),
                )
            )

    return RegressionResult(passed=not issues, issues=issues)
