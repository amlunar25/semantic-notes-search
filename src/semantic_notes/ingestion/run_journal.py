import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from semantic_notes.models import (
    IndexRun,
    IndexRunStatus,
)


class IndexRunJournal:
    """
    Records the status of the current or most recent indexing run.
    """

    def __init__(self, journal_path: Path) -> None:
        self._journal_path = journal_path

    def start(self) -> IndexRun:
        run = IndexRun(
            run_id=str(uuid4()),
            status=IndexRunStatus.RUNNING,
            started_at=datetime.now(UTC),
            completed_at=None,
            current_source=None,
            processed_documents=0,
            error_message=None,
        )

        self.save(run)
        return run

    def mark_processing(
        self,
        run: IndexRun,
        source: str,
    ) -> IndexRun:
        updated_run = replace(
            run,
            current_source=source,
        )

        self.save(updated_run)
        return updated_run

    def mark_document_completed(
        self,
        run: IndexRun,
    ) -> IndexRun:
        updated_run = replace(
            run,
            processed_documents=(run.processed_documents + 1),
            current_source=None,
        )

        self.save(updated_run)
        return updated_run

    def complete(
        self,
        run: IndexRun,
    ) -> IndexRun:
        completed_run = replace(
            run,
            status=IndexRunStatus.COMPLETED,
            completed_at=datetime.now(UTC),
            current_source=None,
        )

        self.save(completed_run)
        return completed_run

    def fail(
        self,
        run: IndexRun,
        error: Exception,
    ) -> IndexRun:
        failed_run = replace(
            run,
            status=IndexRunStatus.FAILED,
            completed_at=datetime.now(UTC),
            error_message=str(error),
        )

        self.save(failed_run)
        return failed_run

    def load(self) -> IndexRun | None:
        if not self._journal_path.exists():
            return None

        raw_data = json.loads(self._journal_path.read_text(encoding="utf-8"))

        return IndexRun(
            run_id=str(raw_data["run_id"]),
            status=IndexRunStatus(raw_data["status"]),
            started_at=datetime.fromisoformat(raw_data["started_at"]),
            completed_at=(
                datetime.fromisoformat(raw_data["completed_at"])
                if raw_data["completed_at"]
                else None
            ),
            current_source=raw_data["current_source"],
            processed_documents=int(raw_data["processed_documents"]),
            error_message=raw_data["error_message"],
        )

    def save(self, run: IndexRun) -> None:
        self._journal_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "run_id": run.run_id,
            "status": run.status.value,
            "started_at": run.started_at.isoformat(),
            "completed_at": (run.completed_at.isoformat() if run.completed_at else None),
            "current_source": run.current_source,
            "processed_documents": (run.processed_documents),
            "error_message": run.error_message,
        }

        temporary_path = self._journal_path.with_suffix(".tmp")

        temporary_path.write_text(
            json.dumps(
                data,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(self._journal_path)
