def evaluate_with_ragas(samples):
    """Optional evaluation adapter. Provide Ragas-compatible samples/metrics."""
    try:
        from ragas import EvaluationDataset, evaluate
    except ImportError as exc:
        raise RuntimeError("Install the 'evaluation' extra to use Ragas.") from exc
    dataset = EvaluationDataset.from_list(samples)
    return evaluate(dataset=dataset)
