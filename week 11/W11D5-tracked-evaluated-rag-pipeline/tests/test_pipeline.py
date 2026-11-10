from tracked_rag.pipeline import run_pipeline, retrieve
from tracked_rag.corpus import QA_PAIRS

def test_ten_rag_samples_have_required_fields():
    samples = run_pipeline(k=2)
    assert len(samples) == 10
    assert all({"user_input", "retrieved_contexts", "response", "reference"} <= set(s) for s in samples)

def test_retrieval_returns_at_most_k_documents():
    assert len(retrieve("refund within 30 days", k=1)) == 1
