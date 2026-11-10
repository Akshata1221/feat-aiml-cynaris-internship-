from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ConversationState(TypedDict, total=False):
    messages: Annotated[list[str], add]
    user_message: str
    intent: str
    reply: str

def classify(state):
    text = state["user_message"].lower()
    intent = "billing" if any(w in text for w in ("refund", "payment", "invoice")) else (
        "technical" if any(w in text for w in ("error", "bug", "broken")) else "general")
    return {"intent": intent, "messages": [f"USER: {state['user_message']}"]}

def route(state): return {"intent": state["intent"]}

def respond(state):
    replies = {
        "billing": "I can help with billing. What is your invoice or transaction reference?",
        "technical": "Let's troubleshoot. What error do you see?",
        "general": "Thanks for the message. How can I help?"
    }
    reply = replies[state["intent"]]
    return {"reply": reply, "messages": [f"ASSISTANT: {reply}"]}

def build_graph(checkpointer=None):
    g = StateGraph(ConversationState)
    g.add_node("classify", classify)
    g.add_node("route", route)
    g.add_node("respond", respond)
    g.add_edge(START, "classify")
    g.add_edge("classify", "route")
    g.add_conditional_edges("route", lambda s: s["intent"], {
        "billing": "respond", "technical": "respond", "general": "respond"})
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())

def build_sqlite_graph(path="checkpoints.sqlite"):
    """Optional durable checkpointing; install langgraph-checkpoint-sqlite."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(path) as saver:
        return build_graph(saver)
