"""Run one Ragas example outside pytest."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from llm_eval_lab.eval_dataset import load_examples
from llm_eval_lab.ragas_concepts import score_record_with_ragas
from llm_eval_lab.toy_rag import build_default_pipeline


class RagasOneCaseDemo:
    """Small script object that shows the minimal Ragas single-sample workflow."""

    async def run(self) -> None:
        """Build one sample and evaluate it with a Ragas metric."""

        load_dotenv()
        pipeline = build_default_pipeline()
        record = pipeline.run_example(load_examples()[0])
        result = await score_record_with_ragas(record)
        print(result)


if __name__ == "__main__":
    asyncio.run(RagasOneCaseDemo().run())

