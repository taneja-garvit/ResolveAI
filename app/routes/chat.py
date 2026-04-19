from fastapi import APIRouter
from app.services.agent_service import run_agent

from fastapi.responses import StreamingResponse
from app.services.agent_service import stream_agent

router = APIRouter()

@router.post("/ask")
def ask(query:ask):
    response = run_agent(query)
    return {"answer": response}

@router.post("/stream-ask")
def stream_ask(query: str):
    generator = stream_agent(query)
    return StreamingResponse(generator(), media_type="text/plain")