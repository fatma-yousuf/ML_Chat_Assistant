from fastapi import APIRouter, HTTPException
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services import agent_service, streaming

router = APIRouter()

@router.post("/", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        response = agent_service.get_chat_response(payload.chat_id, payload.query)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
def chat_stream(payload: ChatRequest):
    return streaming.stream_response(
        lambda: agent_service.get_chat_stream(payload.query, payload.chat_id)
    )