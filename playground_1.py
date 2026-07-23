from pathlib import Path

from semantic_notes.ingestion.document_loader import (
    MarkdownDocumentLoader,
)

loader = MarkdownDocumentLoader()

documents = loader.load_directory(
    Path("data/notes")
)

for document in documents:
    print("TITLE:", document.title)
    print("SOURCE:", document.source_path)
    # print("CONTENT:", document.content)
    print("-" * 50)