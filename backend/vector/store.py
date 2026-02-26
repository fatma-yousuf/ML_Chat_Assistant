# backend/vector/store.py
from __future__ import annotations

from langchain_chroma import Chroma
from backend.vector.embeddings import HFEmbeddings

# NOTE:
# - The "build index" code should live in scripts/build_index.py
# - This module should ONLY load the persisted Chroma DB.

embeddings = HFEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="backend/data/chroma_db",
    embedding_function=embeddings,
    collection_name="hands_on_ml_book",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
