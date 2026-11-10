from pathlib import Path
from .corpus import DOCUMENTS, QA_PAIRS

def retrieve(question, documents=None, k=2):
    docs = documents or DOCUMENTS
    terms = {w.lower().strip("?!.,") for w in question.split() if len(w) > 3}
    ranked = sorted(docs, key=lambda d: sum(t in d.lower() for t in terms), reverse=True)
    return ranked[:k]

def answer(question, contexts):
    # Transparent extractive baseline, designed to run without paid model APIs.
    return contexts[0] if contexts else "I don't have enough information in the knowledge base."

def run_pipeline(k=2):
    results = []
    for question, reference in QA_PAIRS:
        contexts = retrieve(question, k=k)
        results.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer(question, contexts),
            "reference": reference,
        })
    return results

def log_results(results, experiment_name="W11D5-Tracked-RAG"):
    try:
        import mlflow
    except ImportError:
        return False
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"rag-k{2}") as run:
        mlflow.log_param("retrieval_k", 2)
        mlflow.log_param("dataset_size", len(results))
        mlflow.log_metric("answered_samples", sum(bool(x["retrieved_contexts"]) for x in results))
        mlflow.log_dict(results, "evaluation_samples.json")
    return True

if __name__ == "__main__":
    outputs = run_pipeline()
    print(f"Prepared {len(outputs)} RAG evaluation samples.")
    print("MLflow logging enabled:", log_results(outputs))
