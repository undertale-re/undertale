import logging
import os

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


class CodestralSummarizier(PipelineStep):
    """Summarize the given code with a codestral model.

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
    name = "🤖 Codestral"

    def __init__(
        self,
        field: str = "text",
        model_node: str = "c-9-14-1",
        port: str = "12001",
        prompt: str = "Explain what the following code does:",
    ):
        super().__init__()

        self.field = field
        self.addr = f"http://{model_node}:{port}/generate"
        self.prompt = prompt

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        from undertale.datasets.pipeline.summariziers.codestral_utils import post_http_request_outline, get_response_outline, validate_outline, schema

        # if not data:
        #     return

        for document in data:
            with self.track_time():
                if self.field == "text":
                    data = document.text
                else:
                    data = document.metadata[self.field]


                response = post_http_request_outline(prompt=f"{self.prompt}\n{data}",
                                                     addr=self.addr, schema=schema, max_tokens=600)
                output = get_response_outline(response)
                valid, validated_output = validate_outline(output[0])

                document.metadata["codestral_summary"] = validated_output
                print(validated_output)
                yield document

                self.stat_update("summarized")
