from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

class SupportState(TypedDict, total=False):
    user_message: str
    history: Annotated[list[str], add]
    intent: str
    priority: str
    draft: str
    response: str
    status: str

def classify(state):
    text = state["user_message"].lower()
    if any(w in text for w in ("urgent", "outage", "security", "account hacked")):
        return {"intent": "incident", "priority": "high",
                "history": [f"USER: {state['user_message']}"]}
    if any(w in text for w in ("refund", "payment", "invoice", "charged")):
        intent = "billing"
    elif any(w in text for w in ("error", "bug", "crash", "not working")):
        intent = "technical"
    else:
        intent = "general"
    return {"intent": intent, "priority": "normal",
            "history": [f"USER: {state['user_message']}"]}

def choose_path(state):
    return "human" if state["priority"] == "high" else "respond"

def respond(state):
    templates = {
        "billing": "I can help with billing. Please provide your invoice or transaction reference.",
        "technical": "Let's troubleshoot. Please share the error and steps to reproduce it.",
        "general": "Thanks for contacting support. How can I help you today?",
        "incident": "Your issue is marked high priority and will be escalated."
    }
    return {"draft": templates[state["intent"]]}

def human_escalation(state):
    decision = interrupt({
        "type": "priority_escalation",
        "customer_message": state["user_message"],
        "draft": "High-priority issue detected. Confirm escalation or enter an edited response."
    })
    response = "Your high-priority issue has been escalated." if decision == "approve" else str(decision)
    return {"response": response, "status": "human-reviewed",
            "history": [f"ASSISTANT: {response}"]}

def finalize(state):
    return {"response": state["draft"], "status": "automated",
            "history": [f"ASSISTANT: {state['draft']}"]}

def build_graph(checkpointer=None):
    g = StateGraph(SupportState)
    g.add_node("classify", classify)
    g.add_node("respond", respond)
    g.add_node("human_escalation", human_escalation)
    g.add_node("finalize", finalize)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", choose_path, {
        "respond": "respond", "human": "human_escalation"})
    g.add_edge("respond", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("human_escalation", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
