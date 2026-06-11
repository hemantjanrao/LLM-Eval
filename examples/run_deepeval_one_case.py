"""Run one DeepEval example outside pytest."""

from __future__ import annotations

from dotenv import load_dotenv

from llm_eval_lab.deepeval_concepts import assert_record_with_deepeval
from llm_eval_lab.eval_dataset import load_examples
from llm_eval_lab.toy_rag import build_default_pipeline


class DeepEvalOneCaseDemo:
    """Small script object that shows the minimal DeepEval workflow."""

    def run(self) -> None:
        """Build one record and evaluate it with DeepEval metrics."""

        load_dotenv()
        pipeline = build_default_pipeline()
        record = pipeline.run_example(load_examples()[0])
        assert_record_with_deepeval(record)


if __name__ == "__main__":
    DeepEvalOneCaseDemo().run()

