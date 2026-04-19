from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.agent_service import run_agent, stream_agent

router = APIRouter()

@router.post("/ask")
def ask(query: str):
    response = run_agent(query)
    return {"answer": response}

@router.get("/stream-ask")
def stream_ask(query: str):
    generator = stream_agent(query)
    return StreamingResponse(generator(), media_type="text/plain")