from semantic_notes.models import (
    ContextItem,
    RagContext,
)
from semantic_notes.rag.prompt_builder import (
    RagPromptBuilder,
)


def test_prompt_contains_question_and_context() -> None:
    context = RagContext(
        items=(
            ContextItem(
                source="data/notes/spark.md",
                title="Spark",
                chunk_index=0,
                content="Spark uses checkpoints.",
            ),
        ),
        combined_text=("[Source 1]\nSpark uses checkpoints."),
    )

    builder = RagPromptBuilder()

    prompt = builder.build(
        question="How does Spark recover?",
        context=context,
    )

    assert "How does Spark recover?" in prompt
    assert "Spark uses checkpoints." in prompt
    assert "using only the supplied context" in prompt


def test_empty_context_creates_safe_prompt() -> None:
    context = RagContext(
        items=(),
        combined_text="",
    )

    builder = RagPromptBuilder()

    prompt = builder.build(
        question="Unknown question",
        context=context,
    )

    assert "No relevant context was retrieved" in prompt
    assert "Do not invent an answer" in prompt
