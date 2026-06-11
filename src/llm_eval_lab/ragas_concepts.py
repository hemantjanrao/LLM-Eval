"""Ragas adapters for the toy RAG examples."""

from __future__ import annotations

import os
from typing import Any, cast

from llm_eval_lab.compat import install_ragas_vertexai_import_shim
from llm_eval_lab.eval_dataset import EvaluationRunRecord


class RagasConceptFactory:
    """Build Ragas-native samples, datasets, and metrics.

    Ragas centers evaluation around structured samples. This factory keeps the
    conversion visible so you can compare the same underlying data with DeepEval's
    `LLMTestCase` model.
    """

    def build_sample(self, record: EvaluationRunRecord) -> Any:
        """Convert one toy RAG run into a Ragas `SingleTurnSample`."""

        install_ragas_vertexai_import_shim()
        from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=record.example.question,
            response=record.actual_answer,
            retrieved_contexts=record.retrieved_contexts,
            reference=record.example.reference_answer,
            reference_contexts=record.example.reference_contexts,
        )

    def build_dataset(self, records: list[EvaluationRunRecord]) -> Any:
        """Create a Ragas `EvaluationDataset` from multiple records."""

        install_ragas_vertexai_import_shim()
        from ragas import EvaluationDataset

        return EvaluationDataset(samples=[self.build_sample(record) for record in records])

    def build_rag_metrics(
        self,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> list[Any]:
        """Return Ragas RAG metrics configured with a shared evaluator model."""

        install_ragas_vertexai_import_shim()
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        client = _openai_client()
        llm = llm_factory(model_name, client=client)
        embeddings = cast(
            Any,
            embedding_factory("openai", model=embedding_model, client=client),
        )
        return [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ]

    def build_rubric_metric(self, model_name: str = "gpt-4o-mini") -> Any:
        """Return a Ragas rubric metric for answer usefulness."""

        install_ragas_vertexai_import_shim()
        from ragas.llms import llm_factory
        from ragas.metrics.collections import RubricsScoreWithReference

        return RubricsScoreWithReference(
            name="learning_rubric_score",
            rubrics={
                "score_1": "The response is inaccurate or not grounded in the context.",
                "score_2": "The response partially answers the question but misses key evidence.",
                "score_3": "The response is accurate, grounded, concise, and useful.",
            },
            llm=llm_factory(model_name, client=_openai_client()),
        )


async def score_record_with_ragas(
    record: EvaluationRunRecord,
    model_name: str = "gpt-4o-mini",
) -> Any:
    """Score one record with a Ragas rubric metric and return the metric result."""

    factory = RagasConceptFactory()
    sample = factory.build_sample(record)
    metric = factory.build_rubric_metric(model_name=model_name)
    return await metric.single_turn_ascore(sample)


def evaluate_records_with_ragas(
    records: list[EvaluationRunRecord],
    model_name: str = "gpt-4o-mini",
) -> Any:
    """Evaluate a dataset with Ragas and return its framework result object."""

    install_ragas_vertexai_import_shim()
    from ragas import evaluate

    factory = RagasConceptFactory()
    dataset = factory.build_dataset(records)
    metrics = factory.build_rag_metrics(model_name=model_name)
    return evaluate(dataset=dataset, metrics=metrics)


def _openai_client() -> Any:
    """Create the OpenAI client object required by Ragas 0.4 factories.

    The placeholder key supports object construction in API-free smoke checks. CLI
    commands that execute LLM metrics still require a real `OPENAI_API_KEY` before
    making network calls.
    """

    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-learning-placeholder"))
