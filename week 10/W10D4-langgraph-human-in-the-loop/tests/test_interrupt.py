from langgraph.types import Command
from human_review.graph import build_graph

def test_interrupt_pauses_and_resume_approves():
    app = build_graph()
    config = {"configurable": {"thread_id": "review-1"}}
    paused = app.invoke({"request": "Please cancel my subscription"}, config)
    assert paused.get("__interrupt__")
    done = app.invoke(Command(resume="approve"), config)
    assert done["status"] == "approved"
    assert "cancel my subscription" in done["approved_text"]

def test_resume_with_human_edit():
    app = build_graph()
    config = {"configurable": {"thread_id": "review-2"}}
    app.invoke({"request": "I was charged twice"}, config)
    done = app.invoke(Command(resume="We are checking the duplicate charge."), config)
    assert done["status"] == "edited"
    assert done["approved_text"].startswith("We are checking")
