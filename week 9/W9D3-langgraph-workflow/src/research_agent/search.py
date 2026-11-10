"""Search adapter with a deterministic offline fallback."""
from .config import settings
from .models import Source

OFFLINE_SOURCES = [
    Source(title="NIST AI Risk Management Framework", url="https://www.nist.gov/itl/ai-risk-management-framework",
           snippet="A voluntary framework for managing risks across the AI lifecycle."),
    Source(title="OWASP Top 10 for LLM Applications", url="https://owasp.org/www-project-top-10-for-large-language-model-applications/",
           snippet="Community guidance on security risks in applications built with large language models."),
    Source(title="MLflow Documentation", url="https://mlflow.org/docs/latest/",
           snippet="Documentation for tracking experiments and managing the machine-learning lifecycle."),
]

def search_web(query: str, max_results: int | None = None) -> list[Source]:
    """Return live DuckDuckGo results; fall back transparently if unavailable."""
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=max_results or settings.search_max_results)
        sources = [
            Source(title=item.get("title") or "Untitled result",
                   url=item.get("href") or item.get("url") or "",
                   snippet=item.get("body") or "")
            for item in results
            if item.get("href") or item.get("url")
        ]
        if sources:
            return sources
    except Exception:
        # Network failures and rate limits should not crash local development.
        pass
    return OFFLINE_SOURCES[:max_results or settings.search_max_results]
