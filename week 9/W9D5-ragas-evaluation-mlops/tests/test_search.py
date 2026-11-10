import sys
import types
from research_agent.search import search_web

def test_search_returns_sources_on_failure(monkeypatch):
    class BrokenDDGS:
        def __init__(self): pass
        def text(self, *args, **kwargs): raise RuntimeError("network unavailable")
    fake_module = types.ModuleType("duckduckgo_search")
    fake_module.DDGS = BrokenDDGS
    monkeypatch.setitem(sys.modules, "duckduckgo_search", fake_module)
    results = search_web("test", max_results=2)
    assert len(results) == 2
    assert all(item.url.startswith("https://") for item in results)
