from research_agent.pipeline import run_baseline
import pytest

def test_offline_report_has_clear_limitations():
    report = run_baseline("AI in healthcare", use_web=False)
    assert report.mode == "offline-baseline"
    assert report.sources == []
    assert any("No live web search" in item for item in report.limitations)

def test_web_mode_attaches_sources(monkeypatch):
    from research_agent import pipeline
    from research_agent.models import Source
    monkeypatch.setattr(pipeline, "search_web", lambda topic: [
        Source(title="Example", url="https://example.com", snippet="Evidence")
    ])
    report = pipeline.run_baseline("AI governance", use_web=True)
    assert len(report.sources) == 1
    assert "https://example.com" in report.to_markdown()

def test_empty_topic_rejected():
    with pytest.raises(ValueError):
        run_baseline("   ")
