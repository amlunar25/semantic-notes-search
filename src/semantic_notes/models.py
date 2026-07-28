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
    index_signature: str
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


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    query: str
    expected_sources: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    query: str
    expected_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    first_relevant_rank: int | None
    hit: bool
    recall: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationSummary:
    total_cases: int
    successful_cases: int
    hit_rate: float
    mean_recall: float
    mean_reciprocal_rank: float
    case_results: tuple[EvaluationCaseResult, ...]