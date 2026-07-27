from pathlib import Path


def normalize_source_path(path: Path) -> str:
    """
    Return a consistent POSIX-style path string.

    Example:
        data\\notes\\spark.md
    becomes:
        data/notes/spark.md
    """

    return path.as_posix()
