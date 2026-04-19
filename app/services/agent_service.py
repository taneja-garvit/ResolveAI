from typing import Generator

from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from app.utils.vector_db import load_vectorstore
from app.services.ml_service import predict_confidence
from app.tools.ticket_tool import create_ticket
from app.tools.refund_tool import process_refund
from app.tools.order_tool import get_order_status

_VECTORSTORE = None


def get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    _VECTORSTORE = load_vectorstore()
    return _VECTORSTORE


def rag_tool(query: str):
    try:
        vectorstore = get_vectorstore()
    except FileNotFoundError:
        return "RAG database not found. Upload a document first at /upload-doc."

    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "I could not find relevant documents for that question."

    context = "\n\n".join(
        f"Source {i + 1}:\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs)
    )

    similarity_score = min(0.99, max(0.0, len(docs) / 3))
    confidence = predict_confidence(similarity_score, len(query))

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

    llm = ChatOpenAI(temperature=0)
    response = llm([HumanMessage(content=prompt)])
    text = response.content.strip() if hasattr(response, "content") else str(response)

    return f"{text}\n\n(confidence: {confidence:.2f})"


def build_tools():
    return [
        Tool(name="RAG", func=rag_tool, description="Answer from documents"),
        Tool(name="Ticket", func=create_ticket, description="Create support ticket"),
        Tool(name="Refund", func=process_refund, description="Process refund"),
        Tool(name="Order", func=get_order_status, description="Get order status"),
    ]


def build_agent(streaming: bool = False):
    llm = ChatOpenAI(streaming=streaming, temperature=0)
    tools = build_tools()

    return initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=False,
    )


def run_agent(query: str):
    try:
        agent = build_agent(streaming=False)
        return agent.run(query)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"Agent error: {exc}"


def stream_agent(query: str) -> Generator[str, None, None]:
    agent = build_agent(streaming=True)
    try:
        response = agent.run(query)
    except FileNotFoundError as exc:
        yield str(exc)
        return
    except Exception as exc:
        yield f"Agent error: {exc}"
        return

    yield response