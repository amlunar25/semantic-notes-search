from pydantic import SecretStr

from semantic_notes.config import Settings
from semantic_notes.llm.factory import (
    create_language_model,
)
from semantic_notes.llm.ollama_model import (
    OllamaLanguageModel,
)
from semantic_notes.llm.openai_model import (
    OpenAILanguageModel,
)


def test_create_ollama_language_model() -> None:
    settings = Settings(
        llm_provider="ollama",
        ollama_host="http://localhost:11434",
        ollama_model="qwen3:8b",
    )

    language_model = create_language_model(settings)

    assert isinstance(
        language_model,
        OllamaLanguageModel,
    )


def test_create_openai_language_model() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key=SecretStr("test-api-key"),
        openai_model="test-model",
    )

    language_model = create_language_model(settings)

    assert isinstance(
        language_model,
        OpenAILanguageModel,
    )
