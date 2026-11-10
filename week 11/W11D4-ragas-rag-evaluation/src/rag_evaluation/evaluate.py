def run_ragas(samples):
    """Ragas API adapter. Metric names follow the assignment; check installed Ragas version."""
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    except ImportError as exc:
        raise RuntimeError("Install the evaluation extra and configure an LLM/embeddings for Ragas.") from exc
    dataset = EvaluationDataset.from_list(samples)
    metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    return evaluate(dataset=dataset, metrics=metrics)
