from langchain.vectorstores import FAISS
import os

DB_PATH = "data/vectorstore"


def save_vectorstore(docs, embeddings):
    os.makedirs(DB_PATH, exist_ok=True)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(DB_PATH)


def load_vectorstore():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {DB_PATH}. Upload documents via /upload-doc first."
        )

    from app.utils.embeddings import get_embeddings
    embeddings = get_embeddings()

    return FAISS.load_local(DB_PATH, embeddings)