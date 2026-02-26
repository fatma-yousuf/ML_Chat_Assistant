# backend/services/agent_service.py
from __future__ import annotations

from backend.services.memory_service import MemoryService
from backend.ai.graph import rag_agent


def get_chat_response(chat_id: str, query: str) -> str:
    mem = MemoryService.get_memory(chat_id)
    mem.append({"role": "user", "content": query})

    result = rag_agent.invoke({"messages": mem})

    # Save assistant reply back to memory
    assistant_text = result["messages"][-1].content
    mem.append({"role": "assistant", "content": assistant_text})
    MemoryService.save_memory(chat_id, mem)

    return assistant_text


def get_chat_stream(query: str, chat_id: str):
    text = get_chat_response(chat_id, query)
    for token in text.split():
        yield token + " "