# backend/services/memory_service.py
from __future__ import annotations

import json
from pathlib import Path


class MemoryService:
    _dir = Path("data/memory")

    @classmethod
    def get_memory(cls, chat_id: str):
        cls._dir.mkdir(parents=True, exist_ok=True)
        path = cls._dir / f"{chat_id}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def save_memory(cls, chat_id: str, mem):
        cls._dir.mkdir(parents=True, exist_ok=True)
        path = cls._dir / f"{chat_id}.json"
        path.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")