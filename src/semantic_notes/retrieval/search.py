from lancedb.db import DBConnection

from semantic_notes.embeddings.encoder import EmbeddingEncoder
from semantic_notes.models import SearchResult


class SemanticSearchService:
    """
    Performs vector similarity searches over indexed note chunks.
    """

    def __init__(
        self,
        database: DBConnection,
        encoder: EmbeddingEncoder,
        table_name: str,
    ) -> None:
        self._database = database
        self._encoder = encoder
        self._table_name = table_name

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("The search query cannot be empty.")

        if limit <= 0:
            raise ValueError("The search limit must be greater than zero.")

        if self._table_name not in self._database.table_names():
            raise RuntimeError(
                f"Table '{self._table_name}' does not exist. Run the index command first."
            )

        query_vector = self._encoder.encode_query(cleaned_query)
        table = self._database.open_table(self._table_name)

        rows = (
            table.search(query_vector)
            .select(
                [
                    "chunk_id",
                    "source",
                    "title",
                    "chunk_index",
                    "content",
                ]
            )
            .limit(limit)
            .to_list()
        )

        return [
            SearchResult(
                chunk_id=str(row["chunk_id"]),
                source=str(row["source"]),
                title=str(row["title"]),
                chunk_index=int(row["chunk_index"]),
                content=str(row["content"]),
                distance=float(row["_distance"]),
            )
            for row in rows
        ]
