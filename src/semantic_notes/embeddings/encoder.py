from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("The search query cannot be empty.")

        return self.encode([cleaned_query])[0]
