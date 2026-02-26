# ML_Chat_Assistant
🤖 ML RAG Project — Hands-On Machine Learning Chatbot
!<img width="1919" height="848" alt="image" src="https://github.com/user-attachments/assets/035875d3-0fab-4dee-b607-5d0c2eabb1bf" />
!<img width="1919" height="849" alt="image" src="https://github.com/user-attachments/assets/f6764b07-8ba5-4de9-a3ec-e565f9eba79e" />

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
<img width="1408" height="768" alt="Gemini_Generated_Image_co0vwdco0vwdco0v" src="https://github.com/user-attachments/assets/cd4549ad-90e3-4e51-a01e-1fdf29ef3dfe" />

## Setup    
1. clone
2. Some basic Git commands are:
```
git clone https://github.com/fatma-yousuf/ml-rag-project.git
cd ml-rag-project
```
3.  Install the necessary libraries
```
pip install -r requirements.txt
```
   
5. Configure environment
   Create a `.env` file in the project root:
```
  GROQ_API_KEY=your_groq_api_key_here
  GROQ_MODEL=qwen/qwen3-32b
```
 
7. Add the book
   Place Hands-On-ML.pdf inside backend/data/
8. Build the vector index
   Required before first run. Chunks the PDF, generates embeddings, and persists to ChromaDB:
```
python scripts/build_index.py
```
   Note: Subsequent starts load the existing index — no need to re-run unless the PDF changes.
9. Start the backend
```
   uvicorn backend.main:app --reload
```
11. Start the frontend
```
   streamlit run frontend/streamlit_app.py
```
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
