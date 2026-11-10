from stateful_graph.graph import classify, route

def test_five_classification_inputs():
    cases = [
        ("Please refund my payment", "billing"),
        ("I was charged twice", "billing"),
        ("The app has an error", "technical"),
        ("The login is not working", "technical"),
        ("What are your support hours?", "general"),
    ]
    for message, expected in cases:
        assert classify({"message": message})["category"] == expected

def test_route_uses_classification():
    assert route({"category": "billing"})["route"] == "billing"
    assert route({"category": "technical"})["route"] == "technical"
    assert route({"category": "general"})["route"] == "general"
