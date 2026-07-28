import pytest

from semantic_notes.evaluation.metrics import (
    calculate_hit,
    calculate_recall,
    calculate_reciprocal_rank,
    find_first_relevant_rank,
)


def test_find_first_relevant_rank() -> None:
    rank = find_first_relevant_rank(
        retrieved_sources=[
            "document-a.md",
            "document-b.md",
            "document-c.md",
        ],
        expected_sources=[
            "document-b.md",
        ],
    )

    assert rank == 2


def test_no_relevant_result_returns_none() -> None:
    rank = find_first_relevant_rank(
        retrieved_sources=[
            "document-a.md",
        ],
        expected_sources=[
            "document-b.md",
        ],
    )

    assert rank is None


def test_hit_is_true_when_expected_document_exists() -> None:
    result = calculate_hit(
        retrieved_sources=[
            "document-a.md",
            "document-b.md",
        ],
        expected_sources=[
            "document-b.md",
        ],
    )

    assert result is True


def test_hit_is_false_when_expected_document_is_missing() -> None:
    result = calculate_hit(
        retrieved_sources=[
            "document-a.md",
        ],
        expected_sources=[
            "document-b.md",
        ],
    )

    assert result is False


def test_recall_for_one_of_two_expected_documents() -> None:
    recall = calculate_recall(
        retrieved_sources=[
            "document-a.md",
            "document-b.md",
        ],
        expected_sources=[
            "document-b.md",
            "document-c.md",
        ],
    )

    assert recall == pytest.approx(0.5)


def test_full_recall() -> None:
    recall = calculate_recall(
        retrieved_sources=[
            "document-a.md",
            "document-b.md",
        ],
        expected_sources=[
            "document-a.md",
            "document-b.md",
        ],
    )

    assert recall == pytest.approx(1.0)


def test_reciprocal_rank_at_position_one() -> None:
    result = calculate_reciprocal_rank(1)

    assert result == pytest.approx(1.0)


def test_reciprocal_rank_at_position_three() -> None:
    result = calculate_reciprocal_rank(3)

    assert result == pytest.approx(1 / 3)


def test_missing_relevant_rank_returns_zero() -> None:
    result = calculate_reciprocal_rank(None)

    assert result == pytest.approx(0.0)
