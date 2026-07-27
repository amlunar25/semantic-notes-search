from dataclasses import dataclass
from pathlib import Path


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

