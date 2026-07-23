import hashlib
import re

from semantic_notes.models import Document, DocumentChunk


class TextChunker:
    """
    Splits documents into overlapping text chunks.

    The implementation tries to preserve paragraph boundaries before
    falling back to character-based splitting for large paragraphs.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for document in documents:
            chunks.extend(self.split_document(document))

        return chunks

    def split_document(
        self,
        document: Document,
    ) -> list[DocumentChunk]:
        document_id = self._create_document_id(document)
        text_chunks = self._split_text(document.content)

        return [
            DocumentChunk(
                chunk_id=self._create_chunk_id(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=content,
                ),
                document_id=document_id,
                source=str(document.source_path),
                title=document.title,
                chunk_index=chunk_index,
                content=content,
            )
            for chunk_index, content in enumerate(text_chunks)
        ]

    def _split_text(self, text: str) -> list[str]:
        cleaned_text = self._clean_text(text)

        if len(cleaned_text) <= self._chunk_size:
            return [cleaned_text]

        paragraphs = re.split(r"\n\s*\n", cleaned_text)
        paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

        chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(paragraph) > self._chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                chunks.extend(self._split_large_text(paragraph))
                continue

            candidate = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph

            if len(candidate) <= self._chunk_size:
                current_chunk = candidate
                continue

            chunks.append(current_chunk.strip())
            current_chunk = self._add_overlap(
                previous_chunk=current_chunk,
                new_text=paragraph,
            )

        if current_chunk:
            chunks.append(current_chunk.strip())

        return self._remove_duplicates_and_empty(chunks)

    def _split_large_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == len(text):
                break

            start = end - self._chunk_overlap

        return chunks

    def _add_overlap(
        self,
        previous_chunk: str,
        new_text: str,
    ) -> str:
        if self._chunk_overlap == 0:
            return new_text

        overlap_text = previous_chunk[-self._chunk_overlap :].strip()

        if not overlap_text:
            return new_text

        return f"{overlap_text}\n\n{new_text}"

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        lines = [line.rstrip() for line in text.splitlines()]

        return "\n".join(lines).strip()

    @staticmethod
    def _create_document_id(document: Document) -> str:
        value = str(document.source_path.resolve())
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _create_chunk_id(
        document_id: str,
        chunk_index: int,
        content: str,
    ) -> str:
        value = f"{document_id}:{chunk_index}:{content}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _remove_duplicates_and_empty(
        chunks: list[str],
    ) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()

        for chunk in chunks:
            normalized_chunk = chunk.strip()

            if not normalized_chunk:
                continue

            if normalized_chunk in seen:
                continue

            seen.add(normalized_chunk)
            results.append(normalized_chunk)

        return results
