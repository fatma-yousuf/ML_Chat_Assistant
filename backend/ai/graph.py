# backend/ai/graph.py
from __future__ import annotations

from typing import TypedDict, List

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from backend.ai.llm import llm
from backend.ai.prompt import SYSTEM_PROMPT
from backend.ai.tools.retriever import tools

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    """LangGraph state: a running list of chat messages (LangChain BaseMessage)."""
    messages: List[BaseMessage]


def call_llm(state: AgentState) -> dict:
    # Always inject the system prompt on every call (keeps behavior consistent)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build graph: LLM -> (maybe tools) -> LLM -> ... -> END
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)

# ToolNode automatically executes tool calls and returns ToolMessage(s)
graph.add_node("tools", ToolNode(tools))

# If the LLM requested tools, go to "tools", otherwise finish.
graph.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})

# After tools run, go back to the LLM to produce a final answer using tool outputs.
graph.add_edge("tools", "llm")

graph.set_entry_point("llm")
rag_agent = graph.compile()
