import argparse
import sys

from semantic_notes.config import settings
from semantic_notes.database.client import connect_database
from semantic_notes.embeddings.encoder import EmbeddingEncoder
from semantic_notes.ingestion.chunker import TextChunker
from semantic_notes.ingestion.document_loader import (
    MarkdownDocumentLoader,
)
from semantic_notes.ingestion.indexer import NotesIndexer
from semantic_notes.retrieval.search import SemanticSearchService


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-notes",
        description="Local semantic search over Markdown notes.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "index",
        help="Load and index Markdown notes.",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed notes.",
    )

    search_parser.add_argument(
        "query",
        type=str,
        help="Natural-language search query.",
    )

    search_parser.add_argument(
        "--limit",
        type=int,
        default=settings.search_limit,
        help="Maximum number of results.",
    )

    subparsers.add_parser(
        "info",
        help="Show the current configuration.",
    )

    return parser


def create_encoder() -> EmbeddingEncoder:
    return EmbeddingEncoder(settings.embedding_model)


def run_index() -> None:
    database = connect_database(settings.lancedb_path)
    encoder = create_encoder()

    loader = MarkdownDocumentLoader()

    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    indexer = NotesIndexer(
        database=database,
        encoder=encoder,
        loader=loader,
        chunker=chunker,
        table_name=settings.lancedb_table,
    )

    indexed_chunks = indexer.index_directory(settings.notes_path)

    print("Indexing completed successfully.")
    print(f"Notes directory: {settings.notes_path}")
    print(f"Database path: {settings.lancedb_path}")
    print(f"Table: {settings.lancedb_table}")
    print(f"Indexed chunks: {indexed_chunks}")


def run_search(query: str, limit: int) -> None:
    database = connect_database(settings.lancedb_path)
    encoder = create_encoder()

    search_service = SemanticSearchService(
        database=database,
        encoder=encoder,
        table_name=settings.lancedb_table,
    )

    results = search_service.search(
        query=query,
        limit=limit,
    )

    print(f'\nSearch query: "{query}"')
    print(f"Results: {len(results)}\n")

    if not results:
        print("No matching notes were found.")
        return

    for position, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result: {position}")
        print(f"Title: {result.title}")
        print(f"Source: {result.source}")
        print(f"Chunk: {result.chunk_index}")
        print(f"Distance: {result.distance:.4f}")
        print("-" * 80)
        print(result.content)
        print()


def run_info() -> None:
    database = connect_database(settings.lancedb_path)

    print("Semantic Notes configuration")
    print(f"Notes path: {settings.notes_path}")
    print(f"Database path: {settings.lancedb_path}")
    print(f"Database tables: {database.table_names()}")
    print(f"Table name: {settings.lancedb_table}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Chunk size: {settings.chunk_size}")
    print(f"Chunk overlap: {settings.chunk_overlap}")
    print(f"Default search limit: {settings.search_limit}")


def main() -> None:
    parser = create_parser()
    arguments = parser.parse_args()

    try:
        if arguments.command == "index":
            run_index()
        elif arguments.command == "search":
            run_search(
                query=arguments.query,
                limit=arguments.limit,
            )
        elif arguments.command == "info":
            run_info()
        else:
            parser.error(f"Unknown command: {arguments.command}")

    except (
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
