import logging
import os

import openai

from .. import transform

logger = logging.getLogger(__name__)


class OpenAISummary(transform.Map):
    """Summarize binary code with an OpenAI model.

    Arguments:
        field: The field to summarize.
        model: The name of the OpenAI model to query.
        prompt: The prompt to include prior to the data to summarize.
    """

    def __init__(
        self,
        field: str,
        model: str = "gpt-3.5-turbo",
        prompt: str = "Explain what the following code does:",
    ):
        self.model = model
        self.prompt = prompt
        self.field = field

        if "OPENAI_API_KEY" not in os.environ:
            message = "missing API key for OpenAI - please set the `OPENAI_API_KEY` environment variable"

            logging.error(message)
            raise EnvironmentError(message)

    def __call__(self, sample):
        data = sample[self.field]

        client = openai.OpenAI()

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": f"{self.prompt}\n{data}"}],
            model=self.model,
        )

        return {"summary": response.choices[0].message.content}
