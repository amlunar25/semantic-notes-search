from semantic_notes.models import SearchResult
from semantic_notes.rag.context_builder import (
    ContextBuilder,
)


def test_build_context_from_search_results() -> None:
    search_results = [
        SearchResult(
            chunk_id="chunk-1",
            source="data/notes/spark.md",
            title="Spark Checkpoints",
            chunk_index=0,
            content="Checkpoints support recovery.",
            distance=0.15,
        )
    ]

    builder = ContextBuilder()

    context = builder.build(search_results)

    assert len(context.items) == 1
    assert context.items[0].source == ("data/notes/spark.md")
    assert "[Source 1]" in context.combined_text
    assert "Checkpoints support recovery." in (context.combined_text)


def test_empty_results_create_empty_context() -> None:
    builder = ContextBuilder()

    context = builder.build([])

    assert context.items == ()
    assert context.combined_text == ""
