def evaluate_with_ragas(samples):
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    except ImportError as exc:
        raise RuntimeError("Install the evaluation extra and configure an LLM/embeddings.") from exc
    return evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    )
