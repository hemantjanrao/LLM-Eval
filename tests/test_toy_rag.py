"""Tests for the deterministic toy RAG pipeline."""

from llm_eval_lab.eval_dataset import load_examples
from llm_eval_lab.toy_rag import build_default_pipeline


class TestToyRagPipeline:
    """Keep the app under evaluation stable enough for learning examples."""

    def test_pipeline_answers_each_example(self) -> None:
        """The toy pipeline should produce an answer and retrieved contexts."""

        pipeline = build_default_pipeline()
        for example in load_examples():
            record = pipeline.run_example(example)
            assert record.actual_answer
            assert record.retrieved_contexts

    def test_pipeline_can_answer_freeform_question(self) -> None:
        """The public answer method should return a plain string."""

        pipeline = build_default_pipeline()
        assert isinstance(pipeline.answer("What is DeepEval?"), str)

