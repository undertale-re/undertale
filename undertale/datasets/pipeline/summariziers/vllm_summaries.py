import logging
import os
from typing import Callable, Optional

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)

class VLLMSummarizer(PipelineStep):
    """Summarizes the given source code with a vllm server.

    Arguments:
        address: the local address of where the vllm server is running

    Input:
        Source code

    Output:
        Adds the fields `source_summary`, to the document metadata. Does not
        modify the `text` field.
    """

    type = "📖 - SUMMARIZER"
    name = "⚙️ - VLLM"

    def __init__(
        self,
        address: str,
        prompt: Optional[str] = "Explain what the following code does:",
        temperature: Optional[float] = .7,
        max_tokens: Optional[int] = 200
    ):
        super().__init__()

        self.address = address
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        from openai import OpenAI

        if not data:
            return

        for document in data:
            with self.track_time():
                code = document.text[:3000]
                client = OpenAI(
                    base_url=f"http://{self.address}:8000/v1",
                    api_key="EMPTY" # vLLM servers typically don't require an API key
                )

                # Example: Create a chat completion
                response = client.chat.completions.create(
                    model="qwencoder", # Specify the model you are serving
                    messages=[
                        {"role": "user", "content": f"{self.prompt}\n{code}"}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                summary = response.choices[0].message.content
                document.metadata["source_summary"] = summary
                yield document

                self.stat_update("summarized")
