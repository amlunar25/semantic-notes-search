from dataclasses import dataclass
from pathlib import Path
from enum import StrEnum
from datetime import datetime


@dataclass(frozen=True)
class Document:
    """
    A complete source document before chunking.
    """

    source_path: Path
    title: str
    content: str


@dataclass(frozen=True)
class DocumentChunk:
    """
    A smaller portion of a document that can be embedded and retrieved.
    """

    chunk_id: str
    document_id: str
    source: str
    title: str
    chunk_index: int
    content: str


@dataclass(frozen=True)
class SearchResult:
    """
    A semantic-search result returned by LanceDB.
    """

    chunk_id: str
    source: str
    title: str
    chunk_index: int
    content: str
    distance: float


@dataclass(frozen=True)
class SimilarityComparison:
    text_a: str
    text_b: str
    similarity: float
    embedding_dimension: int


class DocumentStatus(StrEnum):
    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    DELETED = "deleted"


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    document_id: str
    content_hash: str
    chunk_count: int


@dataclass(frozen=True)
class DocumentChange:
    source: str
    status: DocumentStatus
    previous_entry: ManifestEntry | None = None


@dataclass(frozen=True)
class IndexingResult:
    new_documents: int
    changed_documents: int
    unchanged_documents: int
    deleted_documents: int
    embedded_chunks: int


class IndexRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class IndexRun:
    run_id: str
    status: IndexRunStatus
    started_at: datetime
    completed_at: datetime | None
    current_source: str | None
    processed_documents: int
    error_message: str | None = None
