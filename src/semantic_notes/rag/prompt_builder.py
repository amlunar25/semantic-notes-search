from semantic_notes.models import RagContext


class RagPromptBuilder:
    """
    Builds a grounded question-answering prompt.
    """

    def build(
        self,
        question: str,
        context: RagContext,
    ) -> str:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        if not context.items:
            return self._build_no_context_prompt(normalized_question)

        return (
            "You are a question-answering assistant.\n\n"
            "Answer the question using only the supplied context.\n"
            "Do not use unsupported assumptions.\n"
            "If the context does not contain enough information, "
            "say that the available notes do not provide the answer.\n"
            "Cite supporting sources using [Source N].\n\n"
            "CONTEXT\n"
            "=======\n"
            f"{context.combined_text}\n\n"
            "QUESTION\n"
            "========\n"
            f"{normalized_question}\n\n"
            "ANSWER\n"
            "======\n"
        )

    @staticmethod
    def _build_no_context_prompt(
        question: str,
    ) -> str:
        return (
            "You are a question-answering assistant.\n\n"
            "No relevant context was retrieved.\n"
            "Do not invent an answer.\n"
            "State that the available notes do not provide "
            "enough information.\n\n"
            "QUESTION\n"
            "========\n"
            f"{question}\n\n"
            "ANSWER\n"
            "======\n"
        )
