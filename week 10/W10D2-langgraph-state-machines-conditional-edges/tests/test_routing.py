from routing_machine.graph import build_graph, classify, select_destination

def test_five_inputs_and_expected_routes():
    examples = [
        ("urgent outage affecting everyone", "incident"),
        ("please refund my invoice", "billing"),
        ("application crash on startup", "technical"),
        ("where can I find your office?", "general"),
        ("security issue: account compromised", "incident"),
    ]
    graph = build_graph()
    for message, expected in examples:
        state = {"message": message}
        classified = classify(state)
        assert select_destination(classified) == expected
        output = graph.invoke(state)
        assert output["destination"] == expected
