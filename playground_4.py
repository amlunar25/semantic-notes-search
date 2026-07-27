from semantic_notes.config import settings
from semantic_notes.embeddings.encoder import EmbeddingEncoder

encoder = EmbeddingEncoder(settings.embedding_model)

texts = [
    "Spark processes late events with watermarks.",
    "How does Spark handle delayed records?",
    "Quito is the capital of Ecuador.",
]

vectors = encoder.encode(texts)

for text, vector in zip(texts, vectors, strict=True):
    print(text)
    print(vector[:5])
    print()
