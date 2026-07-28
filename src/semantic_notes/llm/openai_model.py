from openai import APIConnectionError
from openai import APIError
from openai import APITimeoutError
from openai import AuthenticationError
from openai import OpenAI
from openai import RateLimitError

from semantic_notes.llm.base import (
    LanguageModel,
    LanguageModelError,
)


class OpenAILanguageModel(LanguageModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        normalized_api_key = api_key.strip()
        normalized_model = model.strip()

        if not normalized_api_key:
            raise ValueError("OpenAI API key cannot be empty.")

        if not normalized_model:
            raise ValueError("OpenAI model cannot be empty.")

        if timeout_seconds <= 0:
            raise ValueError("Timeout must be greater than zero.")

        self._model = normalized_model

        self._client = OpenAI(
            api_key=normalized_api_key,
            timeout=timeout_seconds,
        )

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise ValueError("LLM prompt cannot be empty.")

        try:
            response = self._client.responses.create(
                model=self._model,
                input=normalized_prompt,
            )

        except AuthenticationError as error:
            raise LanguageModelError(
                "OpenAI authentication failed. Check OPENAI_API_KEY."
            ) from error

        except RateLimitError as error:
            raise LanguageModelError("OpenAI rate limit was reached.") from error

        except APITimeoutError as error:
            raise LanguageModelError("The OpenAI request timed out.") from error

        except APIConnectionError as error:
            raise LanguageModelError("Could not connect to OpenAI.") from error

        except APIError as error:
            raise LanguageModelError(f"OpenAI request failed: {error}") from error

        generated_text = response.output_text.strip()

        if not generated_text:
            raise LanguageModelError("The language model returned an empty response.")

        return generated_text
