import hashlib


def create_index_signature(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    raw_signature = (
        f"embedding_model={embedding_model};chunk_size={chunk_size};chunk_overlap={chunk_overlap}"
    )

    return hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()
