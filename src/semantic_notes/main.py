import argparse
import sys
from pathlib import Path

from semantic_notes.config import settings
from semantic_notes.database.client import connect_database
from semantic_notes.embeddings.encoder import EmbeddingEncoder
from semantic_notes.ingestion.chunker import TextChunker
from semantic_notes.ingestion.document_loader import (
    MarkdownDocumentLoader,
)
from semantic_notes.ingestion.indexer import NotesIndexer
from semantic_notes.retrieval.search import SemanticSearchService
from semantic_notes.embeddings.comparator import (
    TextSimilarityComparator,
)
from semantic_notes.embeddings.similarity import (
    vector_magnitude,
)
from semantic_notes.database.notes_repository import (
    NotesRepository,
)
from semantic_notes.ingestion.change_detector import (
    DocumentChangeDetector,
)
from semantic_notes.ingestion.manifest import (
    ManifestRepository,
)
from semantic_notes.ingestion.run_journal import (
    IndexRunJournal,
)
from semantic_notes.evaluation.dataset import (
    EvaluationDatasetLoader,
)
from semantic_notes.evaluation.evaluator import (
    RetrievalEvaluator,
)
from semantic_notes.models import EvaluationSummary
from semantic_notes.ingestion.index_signature import (
    create_index_signature,
)
from semantic_notes.rag.context_builder import (
    ContextBuilder,
)
from semantic_notes.rag.preparation import (
    RagPreparationService,
)
from semantic_notes.rag.prompt_builder import (
    RagPromptBuilder,
)
from semantic_notes.llm.factory import (
    create_language_model,
)
from semantic_notes.rag.service import RagService


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

    evaluation_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate semantic-search quality.",
    )

    evaluation_parser.add_argument(
        "--dataset",
        type=str,
        default=str(settings.evaluation_dataset_path),
        help="Path to the evaluation JSON dataset.",
    )

    evaluation_parser.add_argument(
        "--limit",
        type=int,
        default=settings.evaluation_limit,
        help="Number of unique documents to evaluate.",
    )

    evaluation_parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Show only failed evaluation cases.",
    )

    evaluation_parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show every evaluation case.",
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

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare the semantic similarity of two texts.",
    )

    subparsers.add_parser(
        "status",
        help="Show the latest indexing run status.",
    )

    compare_parser.add_argument(
        "text_a",
        type=str,
        help="First text.",
    )

    compare_parser.add_argument(
        "text_b",
        type=str,
        help="Second text.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a query embedding and its search results.",
    )

    inspect_parser.add_argument(
        "query",
        type=str,
        help="Query to inspect.",
    )

    inspect_parser.add_argument(
        "--limit",
        type=int,
        default=settings.search_limit,
        help="Maximum number of search results.",
    )

    prepare_rag_parser = subparsers.add_parser(
        "prepare-rag",
        help="Retrieve context and display the RAG prompt.",
    )

    prepare_rag_parser.add_argument(
        "question",
        type=str,
        help="Question to prepare for RAG.",
    )

    prepare_rag_parser.add_argument(
        "--limit",
        type=int,
        default=settings.search_limit,
        help="Maximum number of context chunks.",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a grounded question about indexed notes.",
    )

    ask_parser.add_argument(
        "question",
        type=str,
        help="Question to answer using indexed notes.",
    )

    ask_parser.add_argument(
        "--limit",
        type=int,
        default=settings.search_limit,
        help="Maximum number of context chunks.",
    )

    ask_parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display the retrieved context.",
    )
    return parser


def create_rag_preparation_service() -> RagPreparationService:
    database = connect_database(settings.lancedb_path)

    encoder = create_encoder()

    search_service = SemanticSearchService(
        database=database,
        encoder=encoder,
        table_name=settings.lancedb_table,
    )

    return RagPreparationService(
        search_service=search_service,
        context_builder=ContextBuilder(),
        prompt_builder=RagPromptBuilder(),
    )


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

    notes_repository = NotesRepository(
        database=database,
        table_name=settings.lancedb_table,
    )

    manifest_repository = ManifestRepository(
        manifest_path=settings.manifest_path,
    )

    change_detector = DocumentChangeDetector()

    run_journal = IndexRunJournal(
        journal_path=settings.run_journal_path,
    )

    index_signature = create_index_signature(
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    indexer = NotesIndexer(
        notes_repository=notes_repository,
        manifest_repository=manifest_repository,
        run_journal=run_journal,
        encoder=encoder,
        loader=loader,
        chunker=chunker,
        change_detector=change_detector,
        index_signature=index_signature,
    )

    result = indexer.index_directory(settings.notes_path)

    print("\nIncremental indexing completed.")
    print("=" * 60)
    print(f"New documents:       {result.new_documents}")
    print(f"Changed documents:   {result.changed_documents}")
    print(f"Unchanged documents: {result.unchanged_documents}")
    print(f"Deleted documents:   {result.deleted_documents}")
    print(f"Embedded chunks:     {result.embedded_chunks}")


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

    for position, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)
        print(f"Rank: {position}")
        print(f"Title: {result.title}")
        print(f"Source: {result.source}")
        print(f"Chunk: {result.chunk_index}")
        print(f"Vector distance: {result.distance:.4f}")
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
    print(f"Manifest path: {settings.manifest_path}")
    print(f"Run journal path: {settings.run_journal_path}")


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
        elif arguments.command == "compare":
            run_compare(
                text_a=arguments.text_a,
                text_b=arguments.text_b,
            )
        elif arguments.command == "inspect":
            run_inspect(
                query=arguments.query,
                limit=arguments.limit,
            )
        elif arguments.command == "info":
            run_info()
        elif arguments.command == "status":
            run_status()
        elif arguments.command == "evaluate":
            run_evaluation(
                dataset_path=arguments.dataset,
                limit=arguments.limit,
                show_failures=arguments.show_failures,
                show_all=arguments.show_all,
            )
        elif arguments.command == "prepare-rag":
            run_prepare_rag(
                question=arguments.question,
                limit=arguments.limit,
            )
        elif arguments.command == "ask":
            run_ask(
                question=arguments.question,
                limit=arguments.limit,
                show_context=arguments.show_context,
            )
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


def run_compare(
    text_a: str,
    text_b: str,
) -> None:
    encoder = create_encoder()

    comparator = TextSimilarityComparator(
        encoder=encoder,
    )

    comparison = comparator.compare(
        text_a=text_a,
        text_b=text_b,
    )

    print("\nSemantic similarity comparison")
    print("=" * 80)
    print(f"Text A: {comparison.text_a}")
    print(f"Text B: {comparison.text_b}")
    print(f"Embedding dimension: {comparison.embedding_dimension}")
    print(f"Cosine similarity: {comparison.similarity:.4f}")

    print_similarity_interpretation(comparison.similarity)


def print_similarity_interpretation(
    similarity: float,
) -> None:
    """
    This interpretation is only educational.

    These thresholds should not be used as production
    relevance thresholds without evaluation.
    """

    if similarity >= 0.75:
        interpretation = "The texts appear strongly related."
    elif similarity >= 0.50:
        interpretation = "The texts appear moderately related."
    elif similarity >= 0.25:
        interpretation = "The texts may share some meaning."
    else:
        interpretation = "The texts appear weakly related."

    print(f"Interpretation: {interpretation}")


def run_inspect(
    query: str,
    limit: int,
) -> None:
    encoder = create_encoder()
    query_vector = encoder.encode_query(query)

    print("\nQuery inspection")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Embedding dimension: {len(query_vector)}")
    print(f"Vector magnitude: {vector_magnitude(query_vector):.6f}")
    print("First 10 embedding values:")

    for index, value in enumerate(query_vector[:10]):
        print(f"  dimension {index}: {value:.6f}")

    print("\nSearching LanceDB...")
    run_search(
        query=query,
        limit=limit,
    )


def run_status() -> None:
    journal = IndexRunJournal(
        journal_path=settings.run_journal_path,
    )

    run = journal.load()

    if run is None:
        print("No indexing run has been recorded.")
        return

    print("\nLatest indexing run")
    print("=" * 60)
    print(f"Run ID: {run.run_id}")
    print(f"Status: {run.status.value}")
    print(f"Started: {run.started_at.isoformat()}")
    print(f"Completed: {run.completed_at.isoformat() if run.completed_at else '-'}")
    print(f"Processed documents: {run.processed_documents}")
    print(f"Current source: {run.current_source or '-'}")
    print(f"Error: {run.error_message or '-'}")


def run_evaluation(
    dataset_path: str,
    limit: int,
    show_failures: bool,
    show_all: bool,
) -> None:
    database = connect_database(settings.lancedb_path)

    encoder = create_encoder()

    search_service = SemanticSearchService(
        database=database,
        encoder=encoder,
        table_name=settings.lancedb_table,
    )

    dataset_loader = EvaluationDatasetLoader()

    cases = dataset_loader.load(Path(dataset_path))

    evaluator = RetrievalEvaluator(search_service=search_service)

    summary = evaluator.evaluate(
        cases=cases,
        limit=limit,
    )

    print_evaluation_summary(
        summary=summary,
        limit=limit,
    )

    if show_all:
        print_evaluation_cases(
            summary=summary,
            only_failures=False,
        )

    elif show_failures:
        print_evaluation_cases(
            summary=summary,
            only_failures=True,
        )


def print_evaluation_summary(
    summary: EvaluationSummary,
    limit: int,
) -> None:
    print("\nRetrieval evaluation")
    print("=" * 70)
    print(f"Evaluation cases: {summary.total_cases}")
    print(f"Successful cases: {summary.successful_cases}")
    print(f"Hit Rate@{limit}: {summary.hit_rate:.4f}")
    print(f"Mean Recall@{limit}: {summary.mean_recall:.4f}")
    print(f"Mean Reciprocal Rank: {summary.mean_reciprocal_rank:.4f}")


def print_evaluation_cases(
    summary: EvaluationSummary,
    only_failures: bool,
) -> None:
    displayed_results = [
        result for result in summary.case_results if not only_failures or not result.hit
    ]

    if not displayed_results:
        if only_failures:
            print("\nNo failed evaluation cases.")
        return

    print("\nEvaluation details")

    for result in displayed_results:
        print("=" * 70)
        print(f"Case: {result.case_id}")
        print(f"Query: {result.query}")
        print(f"Hit: {'yes' if result.hit else 'no'}")
        print(f"Recall: {result.recall:.4f}")
        print(f"First relevant rank: {result.first_relevant_rank or '-'}")
        print(f"Reciprocal rank: {result.reciprocal_rank:.4f}")

        print("Expected sources:")

        for source in result.expected_sources:
            print(f"  - {source}")

        print("Retrieved sources:")

        for rank, source in enumerate(
            result.retrieved_sources,
            start=1,
        ):
            marker = "✓" if source in result.expected_sources else " "

            print(f"  {rank}. [{marker}] {source}")


def run_prepare_rag(
    question: str,
    limit: int,
) -> None:
    preparation_service = create_rag_preparation_service()

    prepared_request = preparation_service.prepare(
        question=question,
        limit=limit,
    )

    print("\nRetrieved sources")
    print("=" * 70)

    for position, item in enumerate(
        prepared_request.context.items,
        start=1,
    ):
        print(f"{position}. {item.title} ({item.source}, chunk {item.chunk_index})")

    print("\nGenerated prompt")
    print("=" * 70)
    print(prepared_request.prompt)


def run_ask(
    question: str,
    limit: int,
    show_context: bool,
) -> None:
    preparation_service = create_rag_preparation_service()

    language_model = create_language_model(settings)

    rag_service = RagService(
        preparation_service=preparation_service,
        language_model=language_model,
    )

    result = rag_service.answer(
        question=question,
        limit=limit,
    )

    print("\nAnswer")
    print("=" * 70)
    print(result.answer)

    if result.sources:
        print("\nSources")
        print("=" * 70)

        for position, source in enumerate(
            result.sources,
            start=1,
        ):
            print(f"{position}. {source}")

    if show_context:
        print("\nRetrieved context")
        print("=" * 70)
        print(result.context.combined_text or "No context retrieved.")


if __name__ == "__main__":
    main()
