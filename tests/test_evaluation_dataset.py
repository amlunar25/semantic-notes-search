import json
from pathlib import Path

import pytest

from semantic_notes.evaluation.dataset import (
    EvaluationDatasetLoader,
)


def test_load_evaluation_cases(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "cases.json"

    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "spark-test",
                    "query": "How does Spark recover?",
                    "expected_sources": ["data/notes/spark.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    loader = EvaluationDatasetLoader()

    cases = loader.load(dataset_path)

    assert len(cases) == 1
    assert cases[0].case_id == "spark-test"
    assert cases[0].query == ("How does Spark recover?")
    assert cases[0].expected_sources == ("data/notes/spark.md",)


def test_missing_dataset_raises_error(
    tmp_path: Path,
) -> None:
    loader = EvaluationDatasetLoader()

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation dataset not found",
    ):
        loader.load(tmp_path / "missing.json")


def test_duplicate_case_ids_are_rejected(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "cases.json"

    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "duplicate",
                    "query": "First query",
                    "expected_sources": ["first.md"],
                },
                {
                    "case_id": "duplicate",
                    "query": "Second query",
                    "expected_sources": ["second.md"],
                },
            ]
        ),
        encoding="utf-8",
    )

    loader = EvaluationDatasetLoader()

    with pytest.raises(
        ValueError,
        match="case IDs must be unique",
    ):
        loader.load(dataset_path)
