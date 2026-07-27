from pathlib import Path
from typing import Any

from semantic_notes.database.notes_repository import (
    NotesRepository,
)
from semantic_notes.embeddings.encoder import EmbeddingEncoder
from semantic_notes.ingestion.change_detector import (
    DocumentChangeDetector,
)
from semantic_notes.ingestion.chunker import TextChunker
from semantic_notes.ingestion.document_loader import (
    MarkdownDocumentLoader,
)
from semantic_notes.ingestion.fingerprint import (
    calculate_content_hash,
)
from semantic_notes.ingestion.manifest import (
    ManifestRepository,
)
from semantic_notes.ingestion.paths import normalize_source_path
from semantic_notes.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    IndexingResult,
    ManifestEntry,
)


class NotesIndexer:
    """
    Incrementally indexes Markdown notes.

    New documents:
        Embed and insert.

    Changed documents:
        Delete previous chunks, embed, and insert again.

    Unchanged documents:
        Skip.

    Deleted documents:
        Delete previous chunks.
    """

    def __init__(
        self,
        notes_repository: NotesRepository,
        manifest_repository: ManifestRepository,
        encoder: EmbeddingEncoder,
        loader: MarkdownDocumentLoader,
        chunker: TextChunker,
        change_detector: DocumentChangeDetector,
    ) -> None:
        self._notes_repository = notes_repository
        self._manifest_repository = manifest_repository
        self._encoder = encoder
        self._loader = loader
        self._chunker = chunker
        self._change_detector = change_detector

    def index_directory(
        self,
        notes_directory: Path,
    ) -> IndexingResult:
        documents = self._loader.load_directory(notes_directory)

        previous_manifest = self._manifest_repository.load()

        changes = self._change_detector.detect(
            documents=documents,
            previous_manifest=previous_manifest,
        )

        documents_by_source = {
            normalize_source_path(document.source_path): document for document in documents
        }

        next_manifest = dict(previous_manifest)

        new_count = 0
        changed_count = 0
        unchanged_count = 0
        deleted_count = 0
        embedded_chunks = 0

        for change in changes:
            if change.status == DocumentStatus.UNCHANGED:
                unchanged_count += 1
                continue

            if change.status == DocumentStatus.DELETED:
                self._notes_repository.delete_by_source(change.source)

                next_manifest.pop(
                    change.source,
                    None,
                )

                deleted_count += 1
                continue

            document = documents_by_source[change.source]

            if change.status == DocumentStatus.CHANGED:
                self._notes_repository.delete_by_source(change.source)
                changed_count += 1

            elif change.status == DocumentStatus.NEW:
                new_count += 1

            chunks = self._chunker.split_document(document)
            rows = self._create_rows(chunks)

            self._write_rows(rows)

            next_manifest[change.source] = self._create_manifest_entry(
                document=document,
                chunks=chunks,
            )

            embedded_chunks += len(chunks)

        self._manifest_repository.save(next_manifest)

        return IndexingResult(
            new_documents=new_count,
            changed_documents=changed_count,
            unchanged_documents=unchanged_count,
            deleted_documents=deleted_count,
            embedded_chunks=embedded_chunks,
        )

    def _write_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return

        if self._notes_repository.table_exists():
            self._notes_repository.add_rows(rows)
        else:
            self._notes_repository.create_table(rows)

    def _create_rows(
        self,
        chunks: list[DocumentChunk],
    ) -> list[dict[str, Any]]:
        contents = [chunk.content for chunk in chunks]

        vectors = self._encoder.encode(contents)

        rows: list[dict[str, Any]] = []

        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):
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

    @staticmethod
    def _create_manifest_entry(
        document: Document,
        chunks: list[DocumentChunk],
    ) -> ManifestEntry:
        if not chunks:
            raise ValueError("A document must produce at least one chunk.")

        return ManifestEntry(
            source=normalize_source_path(document.source_path),
            document_id=chunks[0].document_id,
            content_hash=calculate_content_hash(document),
            chunk_count=len(chunks),
        )
