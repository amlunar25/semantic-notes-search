from collections.abc import Sequence


def find_first_relevant_rank(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> int | None:
    expected_set = set(expected_sources)

    for rank, source in enumerate(
        retrieved_sources,
        start=1,
    ):
        if source in expected_set:
            return rank

    return None


def calculate_hit(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> bool:
    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)

    return bool(retrieved_set.intersection(expected_set))


def calculate_recall(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    expected_set = set(expected_sources)

    if not expected_set:
        raise ValueError("Expected sources cannot be empty.")

    retrieved_set = set(retrieved_sources)

    relevant_retrieved = retrieved_set.intersection(expected_set)

    return len(relevant_retrieved) / len(expected_set)


def calculate_reciprocal_rank(
    first_relevant_rank: int | None,
) -> float:
    if first_relevant_rank is None:
        return 0.0

    if first_relevant_rank <= 0:
        raise ValueError("Relevant rank must be greater than zero.")

    return 1.0 / first_relevant_rank
