import pytest

from semantic_notes.embeddings.similarity import (
    cosine_similarity,
    dot_product,
    vector_magnitude,
)


def test_dot_product() -> None:
    result = dot_product(
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    )

    assert result == pytest.approx(32.0)


def test_vector_magnitude() -> None:
    result = vector_magnitude(
        [3.0, 4.0]
    )

    assert result == pytest.approx(5.0)


def test_identical_vectors_have_similarity_one() -> None:
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    assert result == pytest.approx(1.0)


def test_perpendicular_vectors_have_similarity_zero() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert result == pytest.approx(-1.0)


def test_vectors_must_have_same_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="same number of dimensions",
    ):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_zero_vector_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="zero-length vector",
    ):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 2.0],
        )