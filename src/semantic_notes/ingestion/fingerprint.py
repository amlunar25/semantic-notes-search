import hashlib

from semantic_notes.models import Document


def calculate_content_hash(document: Document) -> str:
    """
    Create a stable fingerprint from the document content.
    """

    normalized_content = normalize_content(document.content)

    return hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()


def normalize_content(content: str) -> str:
    """
    Normalize line endings and trailing whitespace before hashing.
    """

    normalized_lines = [
        line.rstrip()
        for line in content.replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .splitlines()
    ]

    return "\n".join(normalized_lines).strip()
