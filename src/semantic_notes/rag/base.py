from typing import Protocol

from semantic_notes.rag.preparation import (
    PreparedRagRequest,
)


class RagRequestPreparer(Protocol):
    def prepare(
        self,
        question: str,
        limit: int,
    ) -> PreparedRagRequest: ...
