from langchain.vectorstores import FAISS
import os

DB_PATH = "data/vectorstore"

def save_vectorstore(docs, embeddings):
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(DB_PATH)


def load_vectorstore():
    embeddings = embeddings = None
    from app.utils.embeddings import get_embeddings
    embeddings = get_embeddings()

    return FAISS.load_local(DB_PATH, embeddings)