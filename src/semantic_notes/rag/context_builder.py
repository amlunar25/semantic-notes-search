from semantic_notes.models import (
    ContextItem,
    RagContext,
    SearchResult,
)


class ContextBuilder:
    """
    Converts semantic-search results into structured RAG context.
    """

    def build(
        self,
        search_results: list[SearchResult],
    ) -> RagContext:
        items = tuple(
            ContextItem(
                source=result.source,
                title=result.title,
                chunk_index=result.chunk_index,
                content=result.content,
            )
            for result in search_results
        )

        combined_text = self._combine_items(items)

        return RagContext(
            items=items,
            combined_text=combined_text,
        )

    @staticmethod
    def _combine_items(
        items: tuple[ContextItem, ...],
    ) -> str:
        sections: list[str] = []

        for position, item in enumerate(
            items,
            start=1,
        ):
            section = (
                f"[Source {position}]\n"
                f"Title: {item.title}\n"
                f"Path: {item.source}\n"
                f"Chunk: {item.chunk_index}\n\n"
                f"{item.content}"
            )

            sections.append(section)

        return "\n\n---\n\n".join(sections)
