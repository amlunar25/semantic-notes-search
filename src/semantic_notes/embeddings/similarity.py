import math
from collections.abc import Sequence


def dot_product(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """
    Multiply corresponding values and add the results.

    Example:
        [1, 2] · [3, 4]
        = (1 * 3) + (2 * 4)
        = 11
    """

    validate_vector_dimensions(vector_a, vector_b)

    return sum(
        value_a * value_b
        for value_a, value_b in zip(
            vector_a,
            vector_b,
            strict=True,
        )
    )


def vector_magnitude(vector: Sequence[float]) -> float:
    """
    Calculate the length of a vector.

    Example:
        magnitude([3, 4])
        = sqrt((3 * 3) + (4 * 4))
        = sqrt(25)
        = 5
    """

    return math.sqrt(
        sum(value * value for value in vector)
    )


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.

    The typical range is:
        1.0  -> same direction
        0.0  -> unrelated directions
       -1.0  -> opposite directions
    """

    validate_vector_dimensions(vector_a, vector_b)

    magnitude_a = vector_magnitude(vector_a)
    magnitude_b = vector_magnitude(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError(
            "Cosine similarity cannot be calculated "
            "for a zero-length vector."
        )

    return dot_product(
        vector_a,
        vector_b,
    ) / (magnitude_a * magnitude_b)


def validate_vector_dimensions(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> None:
    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same number of dimensions. "
            f"Received {len(vector_a)} and {len(vector_b)}."
        )

    if not vector_a:
        raise ValueError("Vectors cannot be empty.")