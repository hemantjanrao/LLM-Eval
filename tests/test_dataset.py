"""Tests for the learning dataset."""

from llm_eval_lab.eval_dataset import EvaluationExample, load_examples


class TestEvaluationDataset:
    """Verify that the sample data stays useful for all framework adapters."""

    def test_examples_have_required_fields(self) -> None:
        """Every example should contain a question, reference answer, and context."""

        examples = load_examples()
        assert examples
        assert all(isinstance(example, EvaluationExample) for example in examples)
        assert all(example.question for example in examples)
        assert all(example.reference_answer for example in examples)
        assert all(example.reference_contexts for example in examples)

