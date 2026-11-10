from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class AgentState(TypedDict, total=False):
    message: str
    category: str
    route: str
    response: str
    approval: str

def classify(state: AgentState) -> dict:
    text = state["message"].lower()
    if any(w in text for w in ("refund", "payment", "charged", "invoice")):
        category = "billing"
    elif any(w in text for w in ("broken", "error", "not working", "bug")):
        category = "technical"
    else:
        category = "general"
    return {"category": category}

def route(state: AgentState) -> dict:
    return {"route": state["category"]}

def respond(state: AgentState) -> dict:
    route_name = state["route"]
    drafts = {
        "billing": "I’ll help with the billing issue. Please share the invoice or transaction reference.",
        "technical": "I’ll help troubleshoot. Please share the error message and the steps to reproduce it.",
        "general": "Thanks for reaching out. I can help with your general question."
    }
    approval = interrupt({"kind": "review_response", "draft": drafts[route_name],
                          "prompt": "Approve this response or provide replacement text."})
    final_text = drafts[route_name] if approval is True or approval == "approve" else str(approval)
    return {"response": final_text, "approval": str(approval)}

def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "route")
    builder.add_conditional_edges("route", lambda s: s["route"],
        {"billing": "respond", "technical": "respond", "general": "respond"})
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())

def run_with_approval(message: str, thread_id: str, approval):
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    paused = graph.invoke({"message": message}, config)
    # First invocation pauses at interrupt; approval is supplied on resume.
    return graph.invoke(Command(resume=approval), config)
