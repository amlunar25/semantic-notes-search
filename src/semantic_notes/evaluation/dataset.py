import json
from pathlib import Path
from typing import Any

from semantic_notes.models import EvaluationCase


class EvaluationDatasetLoader:
    """
    Loads retrieval evaluation cases from a JSON file.
    """

    def load(
        self,
        dataset_path: Path,
    ) -> list[EvaluationCase]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

        if not dataset_path.is_file():
            raise ValueError(f"Evaluation dataset path is not a file: {dataset_path}")

        raw_content = dataset_path.read_text(encoding="utf-8").strip()

        if not raw_content:
            raise ValueError("Evaluation dataset cannot be empty.")

        raw_cases: list[dict[str, Any]] = json.loads(raw_content)

        if not isinstance(raw_cases, list):
            raise ValueError("Evaluation dataset must contain a JSON list.")

        cases = [self._parse_case(raw_case) for raw_case in raw_cases]

        if not cases:
            raise ValueError("Evaluation dataset must contain at least one case.")

        self._validate_unique_case_ids(cases)

        return cases

    @staticmethod
    def _parse_case(
        raw_case: dict[str, Any],
    ) -> EvaluationCase:
        case_id = str(raw_case.get("case_id", "")).strip()

        query = str(raw_case.get("query", "")).strip()

        raw_expected_sources = raw_case.get(
            "expected_sources",
            [],
        )

        if not case_id:
            raise ValueError("Every evaluation case requires a case_id.")

        if not query:
            raise ValueError(f"Evaluation case '{case_id}' requires a query.")

        if not isinstance(
            raw_expected_sources,
            list,
        ):
            raise ValueError(f"Evaluation case '{case_id}' must use a list for expected_sources.")

        expected_sources = tuple(
            str(source).strip() for source in raw_expected_sources if str(source).strip()
        )

        if not expected_sources:
            raise ValueError(f"Evaluation case '{case_id}' requires at least one expected source.")

        return EvaluationCase(
            case_id=case_id,
            query=query,
            expected_sources=expected_sources,
        )

    @staticmethod
    def _validate_unique_case_ids(
        cases: list[EvaluationCase],
    ) -> None:
        case_ids = [case.case_id for case in cases]

        unique_case_ids = set(case_ids)

        if len(case_ids) != len(unique_case_ids):
            raise ValueError("Evaluation case IDs must be unique.")
