"""Command-line interface for exploring the learning project."""

from __future__ import annotations

import asyncio
import os

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from llm_eval_lab.deepeval_concepts import assert_record_with_deepeval
from llm_eval_lab.eval_dataset import EvaluationRunRecord, load_examples
from llm_eval_lab.local_metrics import score_record
from llm_eval_lab.ragas_concepts import score_record_with_ragas
from llm_eval_lab.toy_rag import build_default_pipeline


class ConsoleReporter:
    """Render evaluation examples and metric results in a learner-friendly format.

    Keeping presentation separate from scoring makes the CLI easier to extend. You
    can swap this class for JSON output later without changing the evaluation code.
    """

    def __init__(self, console: Console) -> None:
        """Attach the reporter to a Rich console."""

        self.console = console

    def records_table(self, records: list[EvaluationRunRecord]) -> None:
        """Print questions, generated answers, and retrieved context counts."""

        table = Table(title="Evaluation Records")
        table.add_column("Question")
        table.add_column("Actual Answer")
        table.add_column("Contexts")
        for record in records:
            table.add_row(
                record.question,
                record.actual_answer,
                str(len(record.retrieved_contexts)),
            )
        self.console.print(table)

    def local_scores_table(self, records: list[EvaluationRunRecord]) -> None:
        """Print deterministic metric scores for all records."""

        table = Table(title="Local Metric Scores")
        table.add_column("Question")
        table.add_column("Metric")
        table.add_column("Score")
        table.add_column("Passed")
        table.add_column("Reason")
        for record in records:
            for result in score_record(record):
                table.add_row(
                    record.question,
                    result.name,
                    f"{result.score:.3f}",
                    "yes" if result.passed else "no",
                    result.reason,
                )
        self.console.print(table)


app = typer.Typer(help="Learn DeepEval and Ragas concepts with a tiny RAG lab.")
console = Console()


def _run_records() -> list[EvaluationRunRecord]:
    """Run all examples through the default toy pipeline."""

    pipeline = build_default_pipeline()
    return [pipeline.run_example(example) for example in load_examples()]


@app.command()
def ask(question: str) -> None:
    """Ask the toy RAG pipeline a question and print its deterministic answer."""

    pipeline = build_default_pipeline()
    console.print(pipeline.answer(question))


@app.command("inspect-dataset")
def inspect_dataset() -> None:
    """Show the examples after they have been run through the toy RAG app."""

    ConsoleReporter(console).records_table(_run_records())


@app.command("score-local")
def score_local() -> None:
    """Run API-free local metrics that approximate key RAG evaluation ideas."""

    ConsoleReporter(console).local_scores_table(_run_records())


@app.command("deepeval-smoke")
def deepeval_smoke() -> None:
    """Run one DeepEval assertion to prove the framework wiring works."""

    load_dotenv()
    _require_openai_key()
    record = _run_records()[0]
    assert_record_with_deepeval(record)
    console.print("DeepEval smoke check completed.")


@app.command("ragas-smoke")
def ragas_smoke() -> None:
    """Run one Ragas metric asynchronously to prove the framework wiring works."""

    load_dotenv()
    _require_openai_key()
    record = _run_records()[0]
    result = asyncio.run(score_record_with_ragas(record))
    console.print(result)


def _require_openai_key() -> None:
    """Fail early with a clear message when LLM metrics cannot call a model."""

    if not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("Set OPENAI_API_KEY in your environment or .env file first.")


if __name__ == "__main__":
    app()
