"""Deterministic baseline pipeline for local runs and regression tests."""
from .models import ResearchReport
from .search import search_web

def run_baseline(topic: str, use_web: bool = False) -> ResearchReport:
    topic = topic.strip()
    if not topic:
        raise ValueError("Research topic must not be empty.")
    sources = search_web(topic) if use_web else []
    if use_web:
        summary = (
            f"This report examines {topic}. The search adapter returned "
            f"{len(sources)} source(s); findings below are a structured starting point "
            "and should be checked against the linked source pages before publication."
        )
        findings = [
            f"Use the retrieved sources to establish the current context for {topic}.",
            "Compare claims across independent sources and record publication dates.",
            "Separate verified facts from interpretation and recommendations.",
        ]
        limitations = [
            "Search snippets are discovery aids, not substitutes for reading the source pages.",
            "The deterministic writer does not synthesize or independently verify source claims.",
        ]
        notes = ["Source list attached.", "Human verification is required before external publication."]
        mode = "web-search (deterministic writer)"
    else:
        summary = (
            f"This baseline report outlines a research plan for {topic}. "
            "It uses no external evidence and is intended for comparison with the sourced run."
        )
        findings = [
            f"Define the scope and important terms related to {topic}.",
            "Identify measurable questions and likely stakeholders.",
            "Validate all factual claims with primary sources before drawing conclusions.",
        ]
        limitations = [
            "No live web search was performed.",
            "Findings are generic planning prompts rather than evidence-backed conclusions.",
        ]
        notes = ["Baseline contains no citations; do not treat it as factual research."]
        mode = "offline-baseline"
    return ResearchReport(topic=topic, executive_summary=summary, key_findings=findings,
                          limitations=limitations, sources=sources,
                          review_notes=notes, mode=mode)

def save_report(report: ResearchReport, output_path: str) -> None:
    from pathlib import Path
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
