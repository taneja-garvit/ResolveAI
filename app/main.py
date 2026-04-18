from fastapi import FastAPI
from app.routes import chat, upload, health

app = FastAPI(title = "AI Resolver")

app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(health.router)