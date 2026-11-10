from support_agent.graph import build_graph

def test_normal_intents_route_without_interrupt():
    app = build_graph()
    config = {"configurable": {"thread_id": "normal-1"}}
    cases = [
        ("Please refund my payment", "billing"),
        ("The application crashes", "technical"),
        ("What are your support hours?", "general"),
    ]
    for message, intent in cases:
        result = app.invoke({"user_message": message}, config)
        assert result["intent"] == intent
        assert result["status"] == "automated"
        assert result["response"]

def test_high_priority_pauses_and_resumes():
    app = build_graph()
    config = {"configurable": {"thread_id": "urgent-1"}}
    paused = app.invoke({"user_message": "URGENT security account hacked"}, config)
    assert paused.get("__interrupt__")
    done = app.invoke({"user_message": "ignored on resume"}, config) if False else None
    from langgraph.types import Command
    done = app.invoke(Command(resume="approve"), config)
    assert done["status"] == "human-reviewed"
    assert "escalated" in done["response"]
