from pathlib import Path
from typing import Any

from lancedb.db import DBConnection

from semantic_notes.embeddings.encoder import EmbeddingEncoder
from semantic_notes.ingestion.chunker import TextChunker
from semantic_notes.ingestion.document_loader import (
    MarkdownDocumentLoader,
)
from semantic_notes.models import DocumentChunk


class NotesIndexer:
    """
    Loads, chunks, embeds, and stores Markdown notes in LanceDB.
    """

    def __init__(
        self,
        database: DBConnection,
        encoder: EmbeddingEncoder,
        loader: MarkdownDocumentLoader,
        chunker: TextChunker,
        table_name: str,
    ) -> None:
        self._database = database
        self._encoder = encoder
        self._loader = loader
        self._chunker = chunker
        self._table_name = table_name

    def index_directory(
        self,
        notes_directory: Path,
    ) -> int:
        documents = self._loader.load_directory(notes_directory)

        if not documents:
            raise ValueError(f"No Markdown documents found in {notes_directory}")

        chunks = self._chunker.split_documents(documents)

        if not chunks:
            raise ValueError("No chunks were generated from the documents.")

        rows = self._create_rows(chunks)

        self._database.create_table(
            self._table_name,
            data=rows,
            mode="overwrite",
        )

        return len(rows)

    def _create_rows(
        self,
        chunks: list[DocumentChunk],
    ) -> list[dict[str, Any]]:
        contents = [chunk.content for chunk in chunks]
        vectors = self._encoder.encode(contents)

        if len(chunks) != len(vectors):
            raise RuntimeError(
                "The number of chunks does not match the number of generated embeddings."
            )

        rows: list[dict[str, Any]] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "vector": vector,
                }
            )

        return rows
