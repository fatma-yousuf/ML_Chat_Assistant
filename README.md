# ML_Chat_Assistant
🤖 ML RAG Project — Hands-On Machine Learning Chatbot
A RAG-powered chatbot built on Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron), using LangGraph, ChromaDB, and Groq.
Ask any question about the book. Get a structured answer with exact chapter and page citations — grounded only in the book text.
Features

RAG answers grounded in the actual book — no hallucination
Page & chapter citations on every response
Persistent multi-session memory — named chats that survive server restarts
LangGraph ReAct agent — LLM decides when to retrieve vs. answer directly
Streaming support via /chat/stream endpoint
Clean Streamlit UI with markdown rendering and dark mode
## Architecture

┌──────────────────────────────────────────┐
│           Streamlit Frontend              │
│  Multi-chat · JSON persistence · Markdown│
└──────────────────┬───────────────────────┘
                   │ POST /chat
┌──────────────────▼───────────────────────┐
│            FastAPI Backend                │
│     /chat  ·  /chat/stream  ·  Pydantic  │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│            Agent Service                  │
│   Load memory → Invoke graph → Save      │
└────────┬─────────────────────┬───────────┘
         │                     │
┌────────▼────────┐   ┌────────▼──────────────────┐
│ Memory Service  │   │      LangGraph Agent        │
│ JSON/chat_id    │   │  llm → tools_condition      │
└─────────────────┘   │  → ToolNode → llm → END    │
                      └────────┬──────────────────-─┘
                               │
                  ┌────────────▼─────────────────┐
                  │       ChromaDB               │
                  │  all-MiniLM-L6-v2 embeddings │
                  │  Chunked PDF · Metadata       │
                  └──────────────────────────────┘
                  
## Project Structure
ml_rag_project/
│
├── backend/
│   ├── main.py                     # FastAPI app entry point
│   ├── api/
│   │   └── chat.py                 # /chat and /chat/stream endpoints
│   ├── services/
│   │   ├── agent_service.py        # Orchestrates memory + graph
│   │   ├── memory_service.py       # Load/save JSON memory per chat
│   │   └── streaming.py            # StreamingResponse wrapper
│   ├── ai/
│   │   ├── llm.py                  # Groq ChatGroq initialization
│   │   ├── prompt.py               # System prompt
│   │   ├── graph.py                # LangGraph ReAct agent
│   │   └── tools/
│   │       └── retriever.py        # @tool: retrieve_passages
│   ├── vector/
│   │   ├── embeddings.py           # HFEmbeddings wrapper
│   │   └── store.py                # Chroma client + retriever
|   ├── data/
│   |   |___Hands-On-ML.pdf             # ← Place your PDF here
|   |
│   └── schemas/
│       └── chat.py                 # ChatRequest / ChatResponse
│
├── frontend/
│   └── streamlit_app.py            # Multi-session chat UI
│
├── .env
├── requirements.txt
└── README.md

## Setup    
1. clone
   git clone https://github.com/fatma-yousuf/ml-rag-project.git
   cd ml-rag-project
2.  Install the necessary libraries
   pip install -r requirements.txt
3. Configure environment
   Create a .env file in the project root:
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=qwen/qwen3-32b
4. Add the book
   Place Hands-On-ML.pdf inside backend/data/
5. Build the vector index
   Required before first run. Chunks the PDF, generates embeddings, and persists to ChromaDB:
   python scripts/build_index.py
   Note: Subsequent starts load the existing index — no need to re-run unless the PDF changes.
6. Start the backend
   uvicorn backend.main:app --reload
7. Start the frontend
   streamlit run frontend/streamlit_app.py
### Open http://localhost:8501
## How the Agent Works
class AgentState(TypedDict):
    messages: List[BaseMessage]
    
The graph has two nodes — llm and tools — connected by tools_condition:
If the LLM's response contains tool calls → route to ToolNode → back to llm
If not → END

The system prompt enforces three strict rules:
Answer only from retrieved excerpts
Cite chapter and page for every claim
If not in the book, say so — never invent

## Requirements

Python 3.11
Groq API key qwen
Hands-On-ML.pdf
~2 GB disk space for ChromaDB index and model weights

### "Built by Fatma Yousuf and Mostafa Mohamed as a course project"
