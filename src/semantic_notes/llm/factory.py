from semantic_notes.config import Settings
from semantic_notes.llm.base import LanguageModel
from semantic_notes.llm.ollama_model import (
    OllamaLanguageModel,
)
from semantic_notes.llm.openai_model import (
    OpenAILanguageModel,
)


def create_language_model(
    settings: Settings,
) -> LanguageModel:
    if settings.llm_provider == "ollama":
        return OllamaLanguageModel(
            host=settings.ollama_host,
            model=settings.ollama_model,
            timeout_seconds=(settings.ollama_timeout_seconds),
        )

    if settings.llm_provider == "openai":
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not configured.")

        api_key = settings.openai_api_key.get_secret_value()

        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is empty.")

        return OpenAILanguageModel(
            api_key=api_key,
            model=settings.openai_model,
            timeout_seconds=(settings.openai_timeout_seconds),
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
