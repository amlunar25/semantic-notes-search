from dataclasses import dataclass

from semantic_notes.models import RagContext
from semantic_notes.rag.context_builder import ContextBuilder
from semantic_notes.rag.prompt_builder import RagPromptBuilder
from semantic_notes.retrieval.search import (
    SemanticSearchService,
)


@dataclass(frozen=True)
class PreparedRagRequest:
    question: str
    context: RagContext
    prompt: str


class RagPreparationService:
    """
    Retrieves context and prepares the final LLM prompt.
    """

    def __init__(
        self,
        search_service: SemanticSearchService,
        context_builder: ContextBuilder,
        prompt_builder: RagPromptBuilder,
    ) -> None:
        self._search_service = search_service
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder

    def prepare(
        self,
        question: str,
        limit: int,
    ) -> PreparedRagRequest:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        search_results = self._search_service.search(
            query=normalized_question,
            limit=limit,
        )

        context = self._context_builder.build(search_results)

        prompt = self._prompt_builder.build(
            question=normalized_question,
            context=context,
        )

        return PreparedRagRequest(
            question=normalized_question,
            context=context,
            prompt=prompt,
        )
