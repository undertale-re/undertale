import logging
import os
from typing import Optional

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


class VLLMSummarizer(PipelineStep):
    """Summarizes the given source code with a vllm server.

    Arguments:
        field: The name of the field you would like to summarize.
        model: The model to be used for summarization.
        prompt: The prompt to use for summarization.
        temperature: Model generation temperature.
        maximum_tokens: The maximum number of tokens to generate.

    Input:
        Any data that you would like to summarize.

    Output:
        Adds the field `summary`, to the document metadata. Does not modify the
        `text` field.
    """

    type = "📄 - Summarize"
    name = "🤖 VLLM"

    def __init__(
        self,
        field: str = "text",
        model: str = "qwencoder",
        prompt: Optional[str] = "Explain what the following code does:",
        temperature: Optional[float] = 0.7,
        maximum_tokens: Optional[int] = 200,
    ):
        super().__init__()

        if "VLLM_SERVER_ADDRESS" not in os.environ:
            message = "missing VLLM server address - please set the `VLLM_SERVER_ADDRESS` environment variable to the full URL of a VLLM compatible server (e.g., `http://vllm.local:8000/v1`)"

            logger.error(message)
            raise EnvironmentError(message)

        self.address = os.environ.get("VLLM_SERVER_ADDRESS")

        if "VLLM_API_KEY" not in os.environ:
            logger.warning(
                "missing VLLM API key - it might not be required, but if it is please set the `VLLM_API_KEY` environment variable"
            )

        self.key = os.environ.get("VLLM_API_KEY", "EMPTY")

        self.field = field
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.maximum_tokens = maximum_tokens

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """"""
        from openai import OpenAI

        if not data:
            return

        for document in data:
            with self.track_time():
                if self.field == "text":
                    data = document.text
                else:
                    data = document.metadata[self.field]

                client = OpenAI(base_url=self.address, api_key=self.key)

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": f"{self.prompt}\n{data}"}],
                    temperature=self.temperature,
                    max_tokens=self.maximum_tokens,
                )

                summary = response.choices[0].message.content
                document.metadata["summary"] = summary

                yield document

                self.stat_update("summarized")
