# backend/ai/llm.py
from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("api_key")
if not API_KEY:
    raise ValueError("Missing GROQ API key. Set GROQ_API_KEY (preferred) or api_key in .env")

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
    temperature=float(os.getenv("TEMPERATURE", "0")),
    api_key=API_KEY,
)