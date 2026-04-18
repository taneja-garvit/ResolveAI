from fastapi import APIRouter
from app.services.agent_service import run_agent

router = APIRouter()

@router.post("/ask")
def ask(query:ask):
    response = run_agent(query)
    return {"answer": response}