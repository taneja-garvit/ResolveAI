import os
import re
from typing import Generator, List, Tuple

from langchain_openai import ChatOpenAI

from app.utils.vector_db import load_vectorstore
from app.services.ml_service import predict_confidence
from app.tools.ticket_tool import create_ticket
from app.tools.refund_tool import process_refund
from app.tools.order_tool import get_order_status

_VECTORSTORE = None
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


def get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    _VECTORSTORE = load_vectorstore()
    return _VECTORSTORE


def _distance_to_similarity(distance: float) -> float:
    safe_distance = max(0.0, float(distance))
    return 1.0 / (1.0 + safe_distance)


def retrieve_ranked_documents(query: str, k: int = 3) -> List[Tuple[object, float]]:
    vectorstore = get_vectorstore()

    try:
        matches = vectorstore.similarity_search_with_score(query, k=k)
    except AttributeError:
        docs = vectorstore.similarity_search(query, k=k)
        return [(doc, 0.5) for doc in docs]

    return [(doc, _distance_to_similarity(score)) for doc, score in matches]


def _get_chat_model() -> ChatOpenAI:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise FileNotFoundError(
            "GROQ_API_KEY is missing. Add it to your environment or .env file."
        )
    if groq_api_key.strip() == "your_groq_api_key_here":
        raise FileNotFoundError(
            "GROQ_API_KEY is still a placeholder. Replace it with your real Groq API key."
        )

    return ChatOpenAI(
        model=DEFAULT_GROQ_MODEL,
        temperature=0,
        api_key=groq_api_key,
        base_url=DEFAULT_GROQ_BASE_URL,
    )


def rag_tool(query: str):
    try:
        ranked_docs = retrieve_ranked_documents(query, k=3)
    except FileNotFoundError:
        return "RAG database not found. Upload a document first at /upload-doc."

    docs = [doc for doc, _ in ranked_docs]
    if not docs:
        return "I could not find relevant documents for that question."

    context = "\n\n".join(
        f"Source {i + 1}:\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs)
    )

    avg_similarity = sum(score for _, score in ranked_docs) / len(ranked_docs)
    top_similarity = max(score for _, score in ranked_docs)
    similarity_score = (0.3 * avg_similarity) + (0.7 * top_similarity)
    normalized_query_length = min(len(query), 120)

    try:
        confidence = predict_confidence(similarity_score, normalized_query_length)
    except FileNotFoundError as exc:
        return str(exc)

    prompt = (
        "You are a helpful assistant. Use the retrieved document passages to answer the user's question. "
        "If the answer is not contained in the passages, say you do not have enough information.\n\n"
        "Document passages:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    llm = _get_chat_model()
    response = llm.invoke(prompt)
    text = response.content.strip() if hasattr(response, "content") else str(response)

    if confidence < 0.35:
        return (
            f"Low confidence ({confidence:.2f}). Escalating to human support.\n\n"
            f"Possible answer from retrieved docs: {text}"
        )

    if confidence < 0.55:
        return f"Tentative answer ({confidence:.2f} confidence): {text}"

    return f"{text}\n\n(confidence: {confidence:.2f})"


def _route_query(query: str) -> str:
    lower_query = query.lower()

    if "ticket" in lower_query or "complaint" in lower_query:
        return create_ticket(query)

    refund_action_words = ("process", "initiate", "start", "issue", "create")
    has_order_reference = bool(re.search(r"\b(order|ord|rm)\s*[-:]?\s*\w+\b", lower_query))
    if "refund" in lower_query and (
        any(word in lower_query for word in refund_action_words)
        or has_order_reference
    ):
        return process_refund(query)

    if (
        "order status" in lower_query
        or "track order" in lower_query
        or ("order" in lower_query and "status" in lower_query)
        or "where is my order" in lower_query
    ):
        return get_order_status(query)

    return rag_tool(query)


def run_agent(query: str):
    try:
        return _route_query(query)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        if "invalid_api_key" in str(exc).lower():
            return (
                "Agent error: Invalid GROQ_API_KEY. "
                "If you ran `copy .env.example .env`, replace placeholder values in `.env` and restart backend."
            )
        return f"Agent error: {exc}"


def stream_agent(query: str) -> Generator[str, None, None]:
    yield run_agent(query)