"""Evaluation orchestration for pluggable RAG applications."""

from __future__ import annotations

from typing import Protocol

from llm_eval_lab.config import EvalConfig, load_config, resolve_dataset_path
from llm_eval_lab.eval_dataset import (
    EvaluationExample,
    EvaluationRunRecord,
    load_examples_from_file,
)
from llm_eval_lab.local_metrics import score_record_with_thresholds
from llm_eval_lab.reporting import (
    EvalReport,
    RecordScore,
    build_eval_report,
    config_hash_from_dict,
    metric_result_to_record_score,
)
from llm_eval_lab.toy_rag import build_default_pipeline


class EvaluableApp(Protocol):
    """Application under test that can produce evaluation run records."""

    def run_example(self, example: EvaluationExample) -> EvaluationRunRecord:
        """Run one example and return observed answer and retrieved contexts."""


def build_default_app() -> EvaluableApp:
    """Return the default toy RAG pipeline as the application under test."""

    return build_default_pipeline()


def filter_examples_by_tags(
    examples: list[EvaluationExample],
    tags: list[str],
) -> list[EvaluationExample]:
    """Return examples whose tags intersect the requested filter tags."""

    if not tags:
        return examples
    tag_set = set(tags)
    return [example for example in examples if tag_set.intersection(example.tags)]


class EvalRunner:
    """Run evaluation examples through an app and score them with configured metrics."""

    def __init__(self, config: EvalConfig | None = None) -> None:
        """Attach a config object or load the default eval.yaml."""

        self.config = config or load_config()

    def load_examples(self) -> list[EvaluationExample]:
        """Load and filter examples according to the current config."""

        dataset_path = resolve_dataset_path(self.config)
        examples = load_examples_from_file(dataset_path)
        return filter_examples_by_tags(examples, self.config.tags)

    def run(
        self,
        app: EvaluableApp | None = None,
        *,
        examples: list[EvaluationExample] | None = None,
    ) -> EvalReport:
        """Execute examples through the app and score them with enabled metric backends."""

        app = app or build_default_app()
        selected_examples = examples if examples is not None else self.load_examples()
        records = [app.run_example(example) for example in selected_examples]

        scores: list[RecordScore] = []
        if self.config.metrics.local.enabled:
            thresholds = self.config.metrics.local.thresholds
            for record in records:
                for result in score_record_with_thresholds(record, thresholds):
                    scores.append(metric_result_to_record_score(record, result))

        config_payload = self.config.model_dump()
        return build_eval_report(
            records,
            scores,
            config_hash=config_hash_from_dict(config_payload),
        )


def run_default_eval(config: EvalConfig | None = None) -> EvalReport:
    """Convenience helper for the default app and config."""

    return EvalRunner(config).run()
