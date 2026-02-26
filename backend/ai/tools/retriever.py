from __future__ import annotations
from typing import Optional, List

from langchain_core.tools import tool
from backend.vector.store import retriever

@tool
def retrieve_passages(query: str, chapter: Optional[int] = None, k: int = 8) -> str:
    """
    Retrieve relevant book passages for a query.
    Optionally restrict retrieval to a chapter number if metadata exists.
    """
    search_kwargs = {"k": k}

    # If your chunks have metadata {"chapter": 1}, this filter will work.
    if chapter is not None:
        search_kwargs["filter"] = {"chapter": chapter}

    docs = retriever.get_relevant_documents(query, **search_kwargs)

    if not docs:
        return "NO_RELEVANT_PASSAGES_FOUND"

    out_lines: List[str] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "book")
        page = meta.get("page", meta.get("page_number", ""))
        chap = meta.get("chapter", "")
        label = f"[{i}] source={src}"
        if chap != "":
            label += f", chapter={chap}"
        if page != "":
            label += f", page={page}"
        out_lines.append(label + "\n" + d.page_content.strip())

    return "\n\n---\n\n".join(out_lines)

tools = [retrieve_passages]