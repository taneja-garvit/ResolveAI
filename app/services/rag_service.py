from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.embeddings import get_embeddings
from app.utils.vector_db import save_vectorstore


def process_document(file_path: str):
    # load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        raise ValueError("No readable content found in the uploaded PDF.")

    # split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.split_documents(documents)
    if not docs:
        raise ValueError("Unable to split uploaded PDF into chunks for indexing.")

    # embeddings
    embeddings = get_embeddings()

    # save into vector DB
    save_vectorstore(docs, embeddings)