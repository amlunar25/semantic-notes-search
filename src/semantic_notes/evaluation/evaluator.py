from semantic_notes.evaluation.metrics import (
    calculate_hit,
    calculate_recall,
    calculate_reciprocal_rank,
    find_first_relevant_rank,
)
from semantic_notes.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationSummary,
)
from semantic_notes.retrieval.search import (
    SemanticSearchService,
)
from collections.abc import Iterable

from semantic_notes.evaluation.metrics import (
    calculate_hit,
    calculate_recall,
    calculate_reciprocal_rank,
    find_first_relevant_rank,
)


class RetrievalEvaluator:
    """
    Evaluates semantic-search quality against labeled cases.
    """

    def __init__(
        self,
        search_service: SemanticSearchService,
    ) -> None:
        self._search_service = search_service

    def evaluate(
        self,
        cases: list[EvaluationCase],
        limit: int,
    ) -> EvaluationSummary:
        if not cases:
            raise ValueError("At least one evaluation case is required.")

        if limit <= 0:
            raise ValueError("Evaluation limit must be greater than zero.")

        case_results = tuple(
            self._evaluate_case(
                case=case,
                limit=limit,
            )
            for case in cases
        )

        successful_cases = sum(1 for result in case_results if result.hit)

        total_cases = len(case_results)

        hit_rate = successful_cases / total_cases

        mean_recall = sum(result.recall for result in case_results) / total_cases

        mean_reciprocal_rank = sum(result.reciprocal_rank for result in case_results) / total_cases

        return EvaluationSummary(
            total_cases=total_cases,
            successful_cases=successful_cases,
            hit_rate=hit_rate,
            mean_recall=mean_recall,
            mean_reciprocal_rank=(mean_reciprocal_rank),
            case_results=case_results,
        )

    def _evaluate_case(
        self,
        case: EvaluationCase,
        limit: int,
    ) -> EvaluationCaseResult:
        chunk_search_limit = limit * 4

        search_results = self._search_service.search(
            query=case.query,
            limit=chunk_search_limit,
        )

        retrieved_sources = self._deduplicate_sources(result.source for result in search_results)[
            :limit
        ]

        first_relevant_rank = find_first_relevant_rank(
            retrieved_sources=retrieved_sources,
            expected_sources=case.expected_sources,
        )

        hit = calculate_hit(
            retrieved_sources=retrieved_sources,
            expected_sources=case.expected_sources,
        )

        recall = calculate_recall(
            retrieved_sources=retrieved_sources,
            expected_sources=case.expected_sources,
        )

        reciprocal_rank = calculate_reciprocal_rank(first_relevant_rank)

        return EvaluationCaseResult(
            case_id=case.case_id,
            query=case.query,
            expected_sources=case.expected_sources,
            retrieved_sources=retrieved_sources,
            first_relevant_rank=first_relevant_rank,
            hit=hit,
            recall=recall,
            reciprocal_rank=reciprocal_rank,
        )

    @staticmethod
    def _deduplicate_sources(
        sources: Iterable[str],
    ) -> tuple[str, ...]:
        unique_sources: list[str] = []
        seen_sources: set[str] = set()

        for source in sources:
            source_value = str(source)

            if source_value in seen_sources:
                continue

            seen_sources.add(source_value)
            unique_sources.append(source_value)

        return tuple(unique_sources)
