from semantic_notes.models import (
    ContextItem,
    RagContext,
)
from semantic_notes.rag.preparation import (
    PreparedRagRequest,
)
from semantic_notes.rag.service import RagService

from tests.fakes import FakeLanguageModel


class FakePreparationService:
    def prepare(
        self,
        question: str,
        limit: int,
    ) -> PreparedRagRequest:
        context = RagContext(
            items=(
                ContextItem(
                    source="data/notes/spark.md",
                    title="Spark",
                    chunk_index=0,
                    content=("Spark checkpoints support recovery."),
                ),
            ),
            combined_text=("[Source 1]\nSpark checkpoints support recovery."),
        )

        return PreparedRagRequest(
            question=question,
            context=context,
            prompt=("Answer using the context: Spark checkpoints support recovery."),
        )


def test_rag_service_generates_grounded_answer() -> None:
    language_model = FakeLanguageModel(response=("Spark uses checkpoints for recovery. [Source 1]"))

    service = RagService(
        preparation_service=FakePreparationService(),
        language_model=language_model,
    )

    result = service.answer(
        question="How does Spark recover?",
        limit=5,
    )

    assert result.answer == ("Spark uses checkpoints for recovery. [Source 1]")

    assert result.sources == ("data/notes/spark.md",)

    assert len(language_model.received_prompts) == 1

    assert "checkpoints support recovery" in (language_model.received_prompts[0])


class EmptyPreparationService:
    def prepare(
        self,
        question: str,
        limit: int,
    ) -> PreparedRagRequest:
        context = RagContext(
            items=(),
            combined_text="",
        )

        return PreparedRagRequest(
            question=question,
            context=context,
            prompt="No context.",
        )


def test_no_context_does_not_call_language_model() -> None:
    language_model = FakeLanguageModel()

    service = RagService(
        preparation_service=EmptyPreparationService(),
        language_model=language_model,
    )

    result = service.answer(
        question="What is not in my notes?",
        limit=5,
    )

    assert "do not provide enough information" in (result.answer)

    assert result.sources == ()
    assert result.context.items == ()
    assert language_model.received_prompts == []
