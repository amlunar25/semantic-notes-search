from pathlib import Path

from semantic_notes.ingestion.run_journal import (
    IndexRunJournal,
)
from semantic_notes.models import IndexRunStatus


def test_start_run(
    tmp_path: Path,
) -> None:
    journal = IndexRunJournal(tmp_path / "run.json")

    run = journal.start()

    assert run.status == IndexRunStatus.RUNNING
    assert run.processed_documents == 0
    assert run.current_source is None

    loaded_run = journal.load()

    assert loaded_run == run


def test_mark_document_completed(
    tmp_path: Path,
) -> None:
    journal = IndexRunJournal(tmp_path / "run.json")

    run = journal.start()

    run = journal.mark_processing(
        run,
        "data/notes/spark.md",
    )

    assert run.current_source == ("data/notes/spark.md")

    run = journal.mark_document_completed(run)

    assert run.processed_documents == 1
    assert run.current_source is None


def test_complete_run(
    tmp_path: Path,
) -> None:
    journal = IndexRunJournal(tmp_path / "run.json")

    run = journal.start()
    run = journal.complete(run)

    assert run.status == IndexRunStatus.COMPLETED
    assert run.completed_at is not None


def test_failed_run_records_error(
    tmp_path: Path,
) -> None:
    journal = IndexRunJournal(tmp_path / "run.json")

    run = journal.start()

    error = RuntimeError("Test failure")
    run = journal.fail(run, error)

    assert run.status == IndexRunStatus.FAILED
    assert run.error_message == "Test failure"
    assert run.completed_at is not None
