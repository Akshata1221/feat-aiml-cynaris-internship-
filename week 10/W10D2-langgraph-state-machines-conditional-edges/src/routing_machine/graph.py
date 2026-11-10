from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class TicketState(TypedDict, total=False):
    message: str
    intent: str
    priority: str
    destination: str
    result: str

def classify(state: TicketState) -> dict:
    text = state["message"].lower()
    if any(x in text for x in ("urgent", "outage", "down", "security")):
        return {"intent": "incident", "priority": "high"}
    if any(x in text for x in ("refund", "invoice", "charge")):
        return {"intent": "billing", "priority": "normal"}
    if any(x in text for x in ("error", "crash", "bug")):
        return {"intent": "technical", "priority": "normal"}
    return {"intent": "general", "priority": "normal"}

def select_destination(state: TicketState) -> str:
    if state["intent"] == "incident":
        return "escalate"
    if state["intent"] == "billing":
        return "billing"
    if state["intent"] == "technical":
        return "technical"
    return "general"

def handle_billing(state): return {"destination": "billing", "result": "Billing queue selected."}
def handle_technical(state): return {"destination": "technical", "result": "Technical queue selected."}
def handle_general(state): return {"destination": "general", "result": "General support selected."}
def handle_escalation(state): return {"destination": "incident", "result": "Priority incident escalation selected."}

def build_graph():
    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("billing", handle_billing)
    g.add_node("technical", handle_technical)
    g.add_node("general", handle_general)
    g.add_node("escalate", handle_escalation)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", select_destination, {
        "billing": "billing", "technical": "technical",
        "general": "general", "escalate": "escalate"})
    for node in ("billing", "technical", "general", "escalate"):
        g.add_edge(node, END)
    return g.compile()
