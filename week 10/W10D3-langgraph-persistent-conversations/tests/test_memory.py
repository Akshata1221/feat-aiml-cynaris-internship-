from conversation_memory.graph import build_graph

def test_same_thread_retains_history():
    app = build_graph()
    config = {"configurable": {"thread_id": "customer-42"}}
    first = app.invoke({"user_message": "I need a refund"}, config)
    second = app.invoke({"user_message": "Here is my invoice"}, config)
    assert first["intent"] == "billing"
    assert "USER: I need a refund" in second["messages"]
    assert "USER: Here is my invoice" in second["messages"]

def test_threads_are_isolated():
    app = build_graph()
    a = app.invoke({"user_message": "hello"}, {"configurable": {"thread_id": "A"}})
    b = app.invoke({"user_message": "app error"}, {"configurable": {"thread_id": "B"}})
    assert a["intent"] == "general"
    assert b["intent"] == "technical"
