from semantic_notes.embeddings.encoder import EmbeddingEncoder
from semantic_notes.embeddings.similarity import cosine_similarity
from semantic_notes.models import SimilarityComparison


class TextSimilarityComparator:
    """
    Converts two texts into embeddings and compares them.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
    ) -> None:
        self._encoder = encoder

    def compare(
        self,
        text_a: str,
        text_b: str,
    ) -> SimilarityComparison:
        cleaned_text_a = text_a.strip()
        cleaned_text_b = text_b.strip()

        if not cleaned_text_a:
            raise ValueError("The first text cannot be empty.")

        if not cleaned_text_b:
            raise ValueError("The second text cannot be empty.")

        vectors = self._encoder.encode(
            [
                cleaned_text_a,
                cleaned_text_b,
            ]
        )

        vector_a = vectors[0]
        vector_b = vectors[1]

        similarity = cosine_similarity(
            vector_a,
            vector_b,
        )

        return SimilarityComparison(
            text_a=cleaned_text_a,
            text_b=cleaned_text_b,
            similarity=similarity,
            embedding_dimension=len(vector_a),
        )