from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class ReviewState(TypedDict, total=False):
    request: str
    draft: str
    approved_text: str
    status: str

def draft_response(state):
    return {"draft": f"Draft response to customer: {state['request']}"}

def human_review(state):
    decision = interrupt({
        "type": "human_approval",
        "draft": state["draft"],
        "instructions": "Resume with 'approve' or replacement response text."
    })
    if decision == "approve":
        return {"approved_text": state["draft"], "status": "approved"}
    return {"approved_text": str(decision), "status": "edited"}

def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("draft", draft_response)
    g.add_node("human_review", human_review)
    g.add_edge(START, "draft")
    g.add_edge("draft", "human_review")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=MemorySaver())

def start_review(app, request, thread_id):
    return app.invoke({"request": request},
                      {"configurable": {"thread_id": thread_id}})

def resume_review(app, thread_id, human_input):
    return app.invoke(Command(resume=human_input),
                      {"configurable": {"thread_id": thread_id}})
