"""Command-line interface for exploring the learning project."""

from __future__ import annotations

import asyncio
import os
from enum import Enum
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from llm_eval_lab.config import default_config_path, load_config, repo_root
from llm_eval_lab.deepeval_concepts import assert_record_with_deepeval
from llm_eval_lab.ragas_concepts import score_record_with_ragas
from llm_eval_lab.regression import compare_reports
from llm_eval_lab.reporting import (
    EvalReport,
    load_report,
    write_report_json,
    write_summary_markdown,
)
from llm_eval_lab.runner import EvalRunner

app = typer.Typer(help="Learn DeepEval and Ragas concepts with a tiny RAG lab.")
console = Console()
DEFAULT_BASELINE_PATH = repo_root() / "artifacts" / "baseline.json"


class OutputFormat(str, Enum):
    """Supported CLI output formats for evaluation runs."""

    console = "console"
    json = "json"
    both = "both"


class ConsoleReporter:
    """Render evaluation examples and metric results in a learner-friendly format."""

    def __init__(self, console: Console) -> None:
        """Attach the reporter to a Rich console."""

        self.console = console

    def records_table(self, report: EvalReport) -> None:
        """Print questions, generated answers, and retrieved context counts."""

        table = Table(title="Evaluation Records")
        table.add_column("Question")
        table.add_column("Actual Answer")
        table.add_column("Contexts")
        for record in report.records:
            table.add_row(
                record.question,
                record.actual_answer,
                str(len(record.retrieved_contexts)),
            )
        self.console.print(table)

    def scores_table(self, report: EvalReport) -> None:
        """Print metric scores for all records."""

        table = Table(title="Local Metric Scores")
        table.add_column("Question")
        table.add_column("Metric")
        table.add_column("Score")
        table.add_column("Passed")
        table.add_column("Reason")
        for score in report.scores:
            table.add_row(
                score.question,
                score.metric,
                f"{score.score:.3f}",
                "yes" if score.passed else "no",
                score.reason,
            )
        self.console.print(table)

    def summary(self, report: EvalReport) -> None:
        """Print aggregate pass rate and mean scores."""

        self.console.print(
            f"Pass rate: {report.summary.pass_rate:.1%} "
            f"({report.summary.total_scores} scores across "
            f"{report.summary.total_examples} examples)"
        )
        for metric, value in sorted(report.summary.mean_scores.items()):
            self.console.print(f"  {metric}: {value:.3f}")


class JsonReporter:
    """Write structured evaluation artifacts to disk."""

    def __init__(self, artifacts_dir: Path) -> None:
        """Attach the reporter to an artifacts directory."""

        self.artifacts_dir = artifacts_dir

    def write(self, report: EvalReport) -> Path:
        """Write JSON and markdown summaries for one evaluation run."""

        run_dir = self.artifacts_dir / report.run_id
        report_path = run_dir / "report.json"
        summary_path = run_dir / "summary.md"
        write_report_json(report, report_path)
        write_summary_markdown(report, summary_path)
        return report_path


def _run_eval(config_path: Path | None = None) -> EvalReport:
    """Run the configured evaluation suite."""

    config = load_config(config_path)
    return EvalRunner(config).run()


@app.command()
def ask(question: str) -> None:
    """Ask the toy RAG pipeline a question and print its deterministic answer."""

    from llm_eval_lab.toy_rag import build_default_pipeline

    pipeline = build_default_pipeline()
    console.print(pipeline.answer(question))


@app.command("inspect-dataset")
def inspect_dataset(
    config: Path | None = typer.Option(None, "--config", help="Path to eval.yaml."),  # noqa: B008
) -> None:
    """Show the examples after they have been run through the toy RAG app."""

    report = _run_eval(config)
    ConsoleReporter(console).records_table(report)


@app.command("score-local")
def score_local(
    config: Path | None = typer.Option(None, "--config", help="Path to eval.yaml."),  # noqa: B008
) -> None:
    """Run API-free local metrics that approximate key RAG evaluation ideas."""

    report = _run_eval(config)
    reporter = ConsoleReporter(console)
    reporter.scores_table(report)
    reporter.summary(report)


@app.command("run-eval")
def run_eval(
    config: Path | None = typer.Option(None, "--config", help="Path to eval.yaml."),  # noqa: B008
    output_format: OutputFormat = typer.Option(  # noqa: B008
        OutputFormat.console,
        "--format",
        help="Output format: console, json, or both.",
    ),
) -> None:
    """Run the full evaluation suite and optionally write structured artifacts."""

    config_path = config or default_config_path()
    eval_config = load_config(config_path)
    report = EvalRunner(eval_config).run()

    if output_format in {OutputFormat.console, OutputFormat.both}:
        reporter = ConsoleReporter(console)
        reporter.records_table(report)
        reporter.scores_table(report)
        reporter.summary(report)

    if output_format in {OutputFormat.json, OutputFormat.both}:
        artifacts_dir = repo_root() / eval_config.artifacts_dir
        report_path = JsonReporter(artifacts_dir).write(report)
        console.print(f"Wrote report to {report_path}")


@app.command("compare-baseline")
def compare_baseline(
    report_path: Path = typer.Argument(..., help="Path to a current report.json."),  # noqa: B008
    baseline_path: Path = typer.Option(  # noqa: B008
        DEFAULT_BASELINE_PATH,
        "--baseline",
        help="Path to the baseline report.json.",
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to eval.yaml."),  # noqa: B008
) -> None:
    """Compare a report against the committed baseline and fail on regression."""

    eval_config = load_config(config)
    current = load_report(report_path)
    baseline = load_report(baseline_path)
    result = compare_reports(current, baseline, regression=eval_config.regression)

    if result.passed:
        console.print("Regression check passed.")
        raise typer.Exit(code=0)

    console.print("Regression check failed:")
    for issue in result.issues:
        console.print(f"  - [{issue.kind}] {issue.message}")
    raise typer.Exit(code=1)


@app.command("deepeval-smoke")
def deepeval_smoke() -> None:
    """Run one DeepEval assertion to prove the framework wiring works."""

    load_dotenv()
    _require_openai_key()
    report = _run_eval()
    assert_record_with_deepeval(report.records[0])
    console.print("DeepEval smoke check completed.")


@app.command("ragas-smoke")
def ragas_smoke() -> None:
    """Run one Ragas metric asynchronously to prove the framework wiring works."""

    load_dotenv()
    _require_openai_key()
    report = _run_eval()
    result = asyncio.run(score_record_with_ragas(report.records[0]))
    console.print(result)


def _require_openai_key() -> None:
    """Fail early with a clear message when LLM metrics cannot call a model."""

    if not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("Set OPENAI_API_KEY in your environment or .env file first.")


if __name__ == "__main__":
    app()
