from pydantic import BaseModel

class ChatRequest(BaseModel):
    chat_id: str
    query: str

class ChatResponse(BaseModel):
    response: str
