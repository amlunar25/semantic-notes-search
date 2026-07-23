from pathlib import Path

from semantic_notes.ingestion.chunker import TextChunker
from semantic_notes.ingestion.document_loader import (
    MarkdownDocumentLoader,
)

loader = MarkdownDocumentLoader()

documents = loader.load_directory(
    Path("data/notes")
)

chunker = TextChunker(
    chunk_size=100,
    chunk_overlap=0,
)

chunks = chunker.split_documents(documents)

for chunk in chunks:
    print("DOCUMENT:", chunk.title)
    print("CHUNK INDEX:", chunk.chunk_index)
    print("CHUNK ID:", chunk.chunk_id)
    print("CONTENT:")
    print(chunk.content)
    print("=" * 70)