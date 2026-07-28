from ollama import Client, ResponseError

from semantic_notes.llm.base import (
    LanguageModel,
    LanguageModelError,
)


class LanguageModelError(RuntimeError):
    """
    Raised when language-model generation fails.
    """


class OllamaLanguageModel(LanguageModel):
    """
    Generates responses using a locally running Ollama model.
    """

    def __init__(
        self,
        host: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        normalized_host = host.strip()
        normalized_model = model.strip()

        if not normalized_host:
            raise ValueError("Ollama host cannot be empty.")

        if not normalized_model:
            raise ValueError("Ollama model cannot be empty.")

        if timeout_seconds <= 0:
            raise ValueError("Timeout must be greater than zero.")

        self._model = normalized_model

        self._client = Client(
            host=normalized_host,
            timeout=timeout_seconds,
        )

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise ValueError("LLM prompt cannot be empty.")

        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": normalized_prompt,
                    }
                ],
                stream=False,
                options={
                    "temperature": 0.1,
                },
            )
        except ResponseError as error:
            if error.status_code == 404:
                raise LanguageModelError(
                    f"Ollama model '{self._model}' was not found. Run: ollama pull {self._model}"
                ) from error

            raise LanguageModelError(f"Ollama request failed: {error.error}") from error

        except ConnectionError as error:
            raise LanguageModelError(
                "Could not connect to Ollama. "
                "Make sure the Ollama application "
                "or server is running."
            ) from error

        except Exception as error:
            raise LanguageModelError(f"Unexpected Ollama error: {error}") from error

        generated_text = response.message.content.strip()

        if not generated_text:
            raise LanguageModelError("Ollama returned an empty response.")

        return generated_text
