import os

from langchain_huggingface import HuggingFaceEmbeddings

_EMBEDDINGS = None

def get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    _EMBEDDINGS = HuggingFaceEmbeddings(model_name=embedding_model)
    return _EMBEDDINGS