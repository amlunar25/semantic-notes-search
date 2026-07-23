from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    notes_path: Path = Path("data/notes")

    lancedb_path: Path = Path("storage/lancedb")
    lancedb_table: str = "notes"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = Field(default=500, ge=100)
    chunk_overlap: int = Field(default=75, ge=0)

    search_limit: int = Field(default=5, ge=1, le=50)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        return self


settings = Settings()
