from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI

from app.services.rag_service import process_document
from app.utils.vector_db import load_vectorstore
from app.services.ml_service import predict_confidence


from app.tools.ticket_tool import create_ticket
from app.tools.refund_tool import process_refund
from app.tools.order_tool import get_order_status


from app.services.ml_service import predict_confidence

def rag_tool(query: str):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query)

    # simple similarity proxy (you can improve later)
    similarity_score = 0.8 if docs else 0.3

    # ML confidence
    confidence = predict_confidence(similarity_score, len(query))

    if confidence < 0.5:
        return f"Low confidence ({confidence}). Escalating to human support."

    return f"Answer (confidence {confidence}): {str(docs)}"


def run_agent(query: str):
    llm = ChatOpenAI(temperature=0)

    tools = [
        Tool(name="RAG", func=rag_tool, description="Answer from documents"),
        Tool(name="Ticket", func=create_ticket, description="Create support ticket"),
        Tool(name="Refund", func=process_refund, description="Process refund"),
        Tool(name="Order", func=get_order_status, description="Get order status"),
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent.run(query)