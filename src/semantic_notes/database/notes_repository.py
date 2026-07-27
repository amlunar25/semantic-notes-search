from typing import Any

from lancedb.db import DBConnection


class NotesRepository:
    """
    Provides database operations for note chunks.
    """

    def __init__(
        self,
        database: DBConnection,
        table_name: str,
    ) -> None:
        self._database = database
        self._table_name = table_name

    def table_exists(self) -> bool:
        return self._table_name in self._database.table_names()

    def create_table(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            raise ValueError("Cannot create a table without rows.")

        self._database.create_table(
            self._table_name,
            data=rows,
        )

    def add_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return

        table = self._database.open_table(self._table_name)

        table.add(rows)

    def delete_by_source(
        self,
        source: str,
    ) -> None:
        if not self.table_exists():
            return

        table = self._database.open_table(self._table_name)

        escaped_source = source.replace("'", "''")

        table.delete(f"source = '{escaped_source}'")
