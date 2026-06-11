"""Tests for regression comparison."""

from __future__ import annotations

from pathlib import Path

from llm_eval_lab.regression import compare_reports
from llm_eval_lab.reporting import EvalReport, EvalRunSummary, RecordScore, load_report
from llm_eval_lab.runner import run_default_eval


class TestRegression:
    """Verify baseline comparison behavior."""

    def test_identical_reports_pass(self) -> None:
        """Comparing a report to itself should pass."""

        report = run_default_eval()
        result = compare_reports(report, report)
        assert result.passed is True
        assert not result.issues

    def test_committed_baseline_passes_against_current_run(self) -> None:
        """Current default run should match the committed baseline."""

        baseline = load_report(Path("artifacts/baseline.json"))
        current = run_default_eval()
        result = compare_reports(current, baseline)
        assert result.passed is True

    def test_mean_score_drop_detected(self) -> None:
        """Large mean score drops should fail the regression gate."""

        baseline = run_default_eval()
        degraded_score = RecordScore(
            question="What is DeepEval useful for?",
            metric="keyword_recall",
            score=0.1,
            passed=False,
            reason="degraded",
            backend="local",
        )
        current = EvalReport(
            run_id="test",
            timestamp="2026-01-01T00:00:00+00:00",
            records=baseline.records,
            scores=[degraded_score if s.metric == "keyword_recall" else s for s in baseline.scores],
            summary=EvalRunSummary(
                total_examples=3,
                total_scores=9,
                pass_rate=0.5,
                mean_scores={"keyword_recall": 0.2, "context_precision": 0.475},
            ),
        )
        result = compare_reports(current, baseline)
        assert result.passed is False
        assert any(issue.kind == "mean_score_drop" for issue in result.issues)

    def test_pass_to_fail_detected(self) -> None:
        """Flipping a passing score to failing should be reported."""

        baseline = run_default_eval()
        flipped_scores = [
            RecordScore(
                question=score.question,
                metric=score.metric,
                score=score.score,
                passed=False if score.metric == "keyword_recall" else score.passed,
                reason=score.reason,
                backend=score.backend,
            )
            for score in baseline.scores
        ]
        current = EvalReport(
            run_id="test",
            timestamp="2026-01-01T00:00:00+00:00",
            records=baseline.records,
            scores=flipped_scores,
            summary=baseline.summary,
        )
        result = compare_reports(current, baseline)
        assert result.passed is False
        assert any(issue.kind == "pass_to_fail" for issue in result.issues)
