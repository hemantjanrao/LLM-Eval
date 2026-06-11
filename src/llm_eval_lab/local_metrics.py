"""Deterministic metrics that teach the intuition behind LLM evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from llm_eval_lab.eval_dataset import EvaluationRunRecord
from llm_eval_lab.toy_rag import tokenize


@dataclass(frozen=True)
class MetricResult:
    """A single metric score with an explanation.

    Frameworks like DeepEval and Ragas often return a numeric score plus a reason.
    This local version mirrors that pattern so you can learn how to read evaluation
    results before using paid or slower LLM-as-judge metrics.
    """

    name: str
    score: float
    passed: bool
    reason: str


class LocalMetric(Protocol):
    """Shared interface for deterministic metrics in this learning project."""

    def score(self, record: EvaluationRunRecord) -> MetricResult:
        """Return an explained score for one evaluated RAG record."""


class KeywordRecallMetric:
    """Approximate answer correctness using overlap with the reference answer.

    This is not a replacement for semantic evaluation. It is a deliberately small
    metric that makes the idea of a reference-based score visible and testable.
    """

    def __init__(self, threshold: float = 0.55) -> None:
        """Set the minimum score needed to pass."""

        self.threshold = threshold

    def score(self, record: EvaluationRunRecord) -> MetricResult:
        """Compare important reference terms against the actual answer."""

        reference_terms = _content_terms(record.example.reference_answer)
        actual_terms = set(_content_terms(record.actual_answer))
        if not reference_terms:
            score = 1.0
        else:
            score = len(set(reference_terms).intersection(actual_terms)) / len(
                set(reference_terms)
            )
        return MetricResult(
            name="keyword_recall",
            score=round(score, 3),
            passed=score >= self.threshold,
            reason="Measures how many reference answer terms appear in the generated answer.",
        )


class ContextPrecisionMetric:
    """Approximate whether retrieved contexts are focused on the expected evidence.

    Precision asks: of what we retrieved, how much was useful? A low score usually
    means the retriever is pulling distracting or unrelated chunks.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Set the minimum average context precision required to pass."""

        self.threshold = threshold

    def score(self, record: EvaluationRunRecord) -> MetricResult:
        """Score each retrieved context by overlap with gold reference contexts."""

        reference_terms = set(_content_terms(" ".join(record.example.reference_contexts)))
        if not record.retrieved_contexts:
            score = 0.0
        else:
            scores = [
                _overlap_ratio(_content_terms(context), reference_terms)
                for context in record.retrieved_contexts
            ]
            score = sum(scores) / len(scores)
        return MetricResult(
            name="context_precision",
            score=round(score, 3),
            passed=score >= self.threshold,
            reason="Measures whether retrieved chunks overlap with expected supporting evidence.",
        )


class FaithfulnessHeuristicMetric:
    """Approximate grounding by checking answer terms against retrieved context.

    Faithfulness is usually best judged semantically. This local metric teaches the
    basic shape: claims in the final answer should be supported by retrieved text.
    """

    def __init__(self, threshold: float = 0.45) -> None:
        """Set the minimum grounding score needed to pass."""

        self.threshold = threshold

    def score(self, record: EvaluationRunRecord) -> MetricResult:
        """Compare answer terms with the retrieved context terms."""

        answer_terms = _content_terms(record.actual_answer)
        context_terms = set(_content_terms(" ".join(record.retrieved_contexts)))
        score = _overlap_ratio(answer_terms, context_terms)
        return MetricResult(
            name="faithfulness_heuristic",
            score=round(score, 3),
            passed=score >= self.threshold,
            reason="Measures whether important answer terms are present in retrieved context.",
        )


def score_record(record: EvaluationRunRecord) -> list[MetricResult]:
    """Run the default local metrics for one evaluation record."""

    metrics: list[LocalMetric] = [
        KeywordRecallMetric(),
        ContextPrecisionMetric(),
        FaithfulnessHeuristicMetric(),
    ]
    return [metric.score(record) for metric in metrics]


def _content_terms(text: str) -> list[str]:
    """Remove common stop words so lexical overlap is less noisy."""

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
    return [token for token in tokenize(text) if token not in stop_words]


def _overlap_ratio(items: Iterable[str], reference_terms: set[str]) -> float:
    """Return the fraction of unique items that are present in a reference set."""

    item_set = set(items)
    if not item_set:
        return 0.0
    return len(item_set.intersection(reference_terms)) / len(item_set)
