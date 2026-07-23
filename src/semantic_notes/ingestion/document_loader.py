from pathlib import Path

from semantic_notes.models import Document


class MarkdownDocumentLoader:
    """
    Loads Markdown files from a directory.

    Each Markdown file becomes one Document instance.
    """

    def load_directory(self, directory: Path) -> list[Document]:
        if not directory.exists():
            raise FileNotFoundError(f"The notes directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"The notes path is not a directory: {directory}")

        documents: list[Document] = []

        for file_path in sorted(directory.rglob("*.md")):
            document = self.load_file(file_path)
            documents.append(document)

        return documents

    def load_file(self, file_path: Path) -> Document:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            raise ValueError(f"Document is empty: {file_path}")

        title = self._extract_title(
            content=content,
            fallback=file_path.stem,
        )

        return Document(
            source_path=file_path,
            title=title,
            content=content,
        )

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        """
        Uses the first Markdown H1 heading as the title.

        Example:
            # Spark Watermarks

        becomes:
            Spark Watermarks
        """

        for line in content.splitlines():
            stripped_line = line.strip()

            if stripped_line.startswith("# "):
                return stripped_line.removeprefix("# ").strip()

        return fallback.replace("-", " ").replace("_", " ").title()
