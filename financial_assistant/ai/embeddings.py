"""The single embedding model: queries must be embedded like the indexed content."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBEDDING_MODEL = "models/gemini-embedding-001"
# Must match Vector(768) in models/document.py and models/graph.py. Changing the
# model or the dimension needs a migration and re-embedding every stored vector.
EMBEDDING_DIMENSIONS = 768

embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIMENSIONS
)
