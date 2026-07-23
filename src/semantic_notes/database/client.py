from pathlib import Path

import lancedb
from lancedb.db import DBConnection


def connect_database(database_path: Path) -> DBConnection:
    database_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(database_path))
