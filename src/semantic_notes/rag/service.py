from semantic_notes.llm.base import LanguageModel
from semantic_notes.models import RagAnswer
from semantic_notes.rag.base import RagRequestPreparer
from semantic_notes.rag.preparation import (
    PreparedRagRequest,
)


class RagService:
    def __init__(
        self,
        preparation_service: RagRequestPreparer,
        language_model: LanguageModel,
    ) -> None:
        self._preparation_service = preparation_service
        self._language_model = language_model

    def answer(
        self,
        question: str,
        limit: int,
    ) -> RagAnswer:
        prepared_request = self._preparation_service.prepare(
            question=question,
            limit=limit,
        )

        if not prepared_request.context.items:
            return RagAnswer(
                question=prepared_request.question,
                answer=(
                    "The available notes do not provide enough information to answer this question."
                ),
                sources=(),
                context=prepared_request.context,
            )

        generated_answer = self._language_model.generate(prepared_request.prompt)

        sources = self._extract_unique_sources(prepared_request)

        return RagAnswer(
            question=prepared_request.question,
            answer=generated_answer,
            sources=sources,
            context=prepared_request.context,
        )

    @staticmethod
    def _extract_unique_sources(
        prepared_request: PreparedRagRequest,
    ) -> tuple[str, ...]:
        context = prepared_request.context

        unique_sources: list[str] = []
        seen_sources: set[str] = set()

        for item in context.items:
            if item.source in seen_sources:
                continue

            seen_sources.add(item.source)
            unique_sources.append(item.source)

        return tuple(unique_sources)
