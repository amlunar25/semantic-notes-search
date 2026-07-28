from pathlib import Path

from pydantic import Field, model_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    notes_path: Path = Path("data/notes")

    lancedb_path: Path = Path("storage/lancedb")
    lancedb_table: str = "notes"

    manifest_path: Path = Path("storage/index_manifest.json")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = Field(default=500, ge=100)
    chunk_overlap: int = Field(default=75, ge=0)

    search_limit: int = Field(default=5, ge=1, le=50)

    run_journal_path: Path = Path("storage/index_run.json")

    evaluation_dataset_path: Path = Path("evaluation/retrieval_cases.json")

    evaluation_limit: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None

    openai_model: str = "gpt-5.5"

    openai_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
    )

    llm_provider: Literal["ollama", "openai"] = "ollama"

    ollama_host: str = "http://localhost:11434"

    ollama_model: str = "qwen3:8b"

    ollama_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600,
    )

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        return self


settings = Settings()
