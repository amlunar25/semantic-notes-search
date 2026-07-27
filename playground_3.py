from semantic_notes.config import settings
from semantic_notes.embeddings.encoder import EmbeddingEncoder

encoder = EmbeddingEncoder(settings.embedding_model)

vector = encoder.encode_query("How does Spark handle late data?")

print("Number of values:", len(vector))
print("First 10 values:", vector[:10])
