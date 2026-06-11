"""A tiny RAG application used as the system under test."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from llm_eval_lab.eval_dataset import EvaluationExample, EvaluationRunRecord

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens for simple lexical retrieval and scoring."""

    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Document:
    """A source passage that the retriever can return as evidence.

    In a production RAG system this would usually include many more fields: source
    URI, chunk index, embedding vector, metadata filters, timestamps, and access
    controls. Here we keep only what is necessary for learning evaluation.
    """

    doc_id: str
    text: str
    topic: str


class ToyKnowledgeBase:
    """An in-memory collection of documents for reproducible examples.

    The knowledge base is intentionally tiny and deterministic. That makes metric
    behavior easy to inspect because the same query always sees the same candidate
    documents and rankings.
    """

    def __init__(self, documents: Iterable[Document]) -> None:
        """Store the provided documents as an ordered collection."""

        self._documents = list(documents)

    @property
    def documents(self) -> list[Document]:
        """Return a copy of the available documents to avoid accidental mutation."""

        return list(self._documents)


class ToyRetriever:
    """A simple keyword-overlap retriever.

    Real retrievers often use embeddings, hybrid search, re-rankers, and metadata
    filters. This class uses token overlap so you can reason about contextual
    precision and recall without needing a vector database.
    """

    def __init__(self, knowledge_base: ToyKnowledgeBase) -> None:
        """Attach the retriever to a knowledge base."""

        self.knowledge_base = knowledge_base

    def retrieve(self, query: str, top_k: int = 2) -> list[Document]:
        """Return the highest-overlap documents for a query."""

        query_terms = set(tokenize(query))
        ranked = sorted(
            self.knowledge_base.documents,
            key=lambda document: self._score(query_terms, document),
            reverse=True,
        )
        return ranked[:top_k]

    def _score(self, query_terms: set[str], document: Document) -> int:
        """Count shared tokens between a query and a candidate document."""

        return len(query_terms.intersection(tokenize(document.text)))


class ToyAnswerGenerator:
    """A deterministic answer generator that summarizes retrieved documents.

    The generator does not call an LLM. It exists to make the evaluation loop
    runnable on any machine and to show the inputs that LLM-based metrics inspect:
    user question, retrieved contexts, actual output, and reference answer.
    """

    def answer(self, question: str, documents: list[Document]) -> str:
        """Create a concise answer from the retrieved documents."""

        if not documents:
            return "I do not have enough context to answer that question."

        combined_context = " ".join(document.text for document in documents)
        if "faithfulness" in question.lower():
            return (
                "Faithfulness measures whether an answer is grounded in the retrieved "
                "context and avoids unsupported claims."
            )
        if "ragas" in question.lower():
            return (
                "Ragas turns subjective vibe checks into repeatable evaluation loops "
                "with datasets, metrics, and feedback for LLM applications."
            )
        if "deepeval" in question.lower():
            return (
                "DeepEval is useful for testing LLM apps with pytest-like workflows, "
                "RAG metrics, safety metrics, and custom G-Eval rubrics."
            )
        return f"Based on the retrieved context: {combined_context}"


class ToyRagPipeline:
    """The complete retriever-plus-generator workflow under evaluation.

    Evaluation frameworks often treat the app as a black box. This class gives us
    both black-box behavior through `answer` and inspectable internals through
    `run_example`, which returns retrieved contexts for RAG metrics.
    """

    def __init__(self, retriever: ToyRetriever, generator: ToyAnswerGenerator) -> None:
        """Create a pipeline from independent retrieval and generation components."""

        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> str:
        """Return only the final answer for simple application use."""

        documents = self.retriever.retrieve(question)
        return self.generator.answer(question, documents)

    def run_example(self, example: EvaluationExample) -> EvaluationRunRecord:
        """Run an evaluation example and preserve all fields needed by metrics."""

        documents = self.retriever.retrieve(example.question)
        actual_answer = self.generator.answer(example.question, documents)
        return EvaluationRunRecord(
            example=example,
            actual_answer=actual_answer,
            retrieved_contexts=[document.text for document in documents],
        )


def build_default_pipeline() -> ToyRagPipeline:
    """Create the default toy RAG app used by the CLI, tests, and eval examples."""

    knowledge_base = ToyKnowledgeBase(
        documents=[
            Document(
                doc_id="deepeval-overview",
                topic="deepeval",
                text=(
                    "DeepEval is an LLM evaluation framework that works like pytest "
                    "for LLM apps and supports answer relevancy, faithfulness, "
                    "contextual metrics, safety metrics, and custom G-Eval rubrics."
                ),
            ),
            Document(
                doc_id="ragas-overview",
                topic="ragas",
                text=(
                    "Ragas helps teams move from vibe checks to systematic evaluation "
                    "loops with datasets, metrics, and feedback for LLM applications."
                ),
            ),
            Document(
                doc_id="faithfulness",
                topic="rag",
                text=(
                    "Faithfulness evaluates whether a generated answer is grounded in "
                    "retrieved context and avoids unsupported claims."
                ),
            ),
            Document(
                doc_id="retrieval-quality",
                topic="rag",
                text=(
                    "Context precision and context recall evaluate whether retrieval "
                    "ranked relevant chunks highly and included all required evidence."
                ),
            ),
        ]
    )
    return ToyRagPipeline(ToyRetriever(knowledge_base), ToyAnswerGenerator())
