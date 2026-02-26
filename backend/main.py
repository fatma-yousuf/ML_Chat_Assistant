from fastapi import FastAPI
from backend.api.chat import router as chat_router

app = FastAPI(title="ML RAG API")
app.include_router(chat_router, prefix="/chat", tags=["chat"])

@app.get("/")
def health():
    return {"status": "ok", "message": "Backend is running. Visit /docs"}