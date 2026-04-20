import os
from typing import Generator, List, Tuple

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from app.utils.vector_db import load_vectorstore
from app.services.ml_service import predict_confidence
from app.tools.ticket_tool import create_ticket
from app.tools.refund_tool import process_refund
from app.tools.order_tool import get_order_status

_VECTORSTORE = None
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


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

    similarity_score = sum(score for _, score in ranked_docs) / len(ranked_docs)

    try:
        confidence = predict_confidence(similarity_score, len(query))
    except FileNotFoundError as exc:
        return str(exc)

    if confidence < 0.5:
        return f"Low confidence ({confidence:.2f}). Escalating to human support."

    prompt = (
        "You are a helpful assistant. Use the retrieved document passages to answer the user's question. "
        "If the answer is not contained in the passages, say you do not have enough information.\n\n"
        "Document passages:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    llm = ChatOpenAI(model=DEFAULT_OPENAI_MODEL, temperature=0)
    response = llm.invoke(prompt)
    text = response.content.strip() if hasattr(response, "content") else str(response)

    return f"{text}\n\n(confidence: {confidence:.2f})"


def build_tools():
    return [
        tool(
            "rag_lookup",
            description="Answer questions using uploaded company documents and support policies.",
        )(rag_tool),
        tool(
            "create_support_ticket",
            description="Create a support ticket for a customer issue.",
        )(create_ticket),
        tool(
            "process_customer_refund",
            description="Start a refund workflow for a customer order.",
        )(process_refund),
        tool(
            "check_order_status",
            description="Look up the current status of an order.",
        )(get_order_status),
    ]


def build_agent():
    llm = ChatOpenAI(model=DEFAULT_OPENAI_MODEL, temperature=0)
    return create_agent(
        model=llm,
        tools=build_tools(),
        system_prompt=(
            "You are an AI customer support copilot. Use document retrieval for company policy questions "
            "and call support tools when the user requests an action like a refund, order lookup, or ticket creation."
        ),
        debug=False,
    )


def _extract_agent_text(result) -> str:
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            content = getattr(messages[-1], "content", messages[-1])
            if isinstance(content, list):
                return "".join(
                    item.get("text", str(item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in content
                ).strip()
            return str(content).strip()

    return str(result).strip()


def run_agent(query: str):
    try:
        agent = build_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        return _extract_agent_text(result)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"Agent error: {exc}"


def stream_agent(query: str) -> Generator[str, None, None]:
    yield run_agent(query)