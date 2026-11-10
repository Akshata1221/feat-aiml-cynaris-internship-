"""Optional LangGraph wrapper for an auditable research workflow."""
def build_graph():
    try:
        from typing import TypedDict
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        raise RuntimeError("Install langgraph to use graph mode.") from exc
    from .pipeline import run_baseline

    class State(TypedDict):
        topic: str
        report: str

    def research(state: State):
        report = run_baseline(state["topic"], use_web=True)
        return {"report": report.to_markdown()}

    graph = StateGraph(State)
    graph.add_node("research_and_report", research)
    graph.set_entry_point("research_and_report")
    graph.add_edge("research_and_report", END)
    return graph.compile()

def run_graph(topic: str) -> str:
    return build_graph().invoke({"topic": topic, "report": ""})["report"]
