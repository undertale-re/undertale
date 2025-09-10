"""Summarize with any OpenAI API compatible service."""

import logging
import os

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


class OpenAISummarizer(PipelineStep):
    """Summarize the given code with an OpenAI model.

    Arguments:
        field: The field to summarize - by default, this summarized the `text`
            field; if supplied, summarize the given metadata field.
        model: The model name to use for summarization.
        prompt: The prompt to use for summarization.

    Input:
        Any code that you would like to summarize.

    Output:
        Adds a metadata field called `summary` containing the summary produced
        by the model. Does not modify the `text` field.
    """

    type = "📄 - Summarize"
    name = "🤖 OpenAI"

    def __init__(
        self,
        field: str = "text",
        model: str = "gpt-3.5-turbo",
        prompt: str = "Explain what the following code does:",
    ):
        super().__init__()

        self.field = field
        self.model = model
        self.prompt = prompt

        if "OPENAI_API_KEY" not in os.environ:
            message = "missing API key for OpenAI - please set the `OPENAI_API_KEY` environment variable"

            logging.error(message)
            raise EnvironmentError(message)

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """"""
        import openai

        if not data:
            return

        for document in data:
            with self.track_time():
                if self.field == "text":
                    data = document.text
                else:
                    data = document.metadata[self.field]

                client = openai.OpenAI()

                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": f"{self.prompt}\n{data}"}],
                    model=self.model,
                )

                document.metadata["summary"] = response.choices[0].message.content

                yield document

                self.stat_update("summarized")
