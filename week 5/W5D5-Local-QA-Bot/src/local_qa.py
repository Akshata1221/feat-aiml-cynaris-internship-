import csv, json, time
from pathlib import Path
import chromadb
from .ollama_client import chat, embed

FACTS = ["ChromaDB is a vector database for storing embeddings.", "Ollama runs language models locally.", "A retrieval-augmented system finds relevant passages before generating an answer.", "Cosine similarity compares the direction of two vectors.", "The llama3.2:3b and qwen2.5:3b models can be compared on identical prompts."]
SYSTEM = "You are a careful technical tutor. Answer only from the supplied context. If insufficient, say you do not know."
QUESTIONS = ["What is ChromaDB?", "What does Ollama do?", "What is retrieval augmented generation?", "How does cosine similarity work?", "Why compare two models?"]

def setup(store="chroma_store"):
    col = chromadb.PersistentClient(path=store).get_or_create_collection("qa_facts", metadata={"hnsw:space": "cosine"})
    col.upsert(ids=[str(i) for i in range(len(FACTS))], documents=FACTS, embeddings=[embed(x) for x in FACTS])
    return col

def answer(col, question, model="llama3.2:3b"):
    hits = col.query(query_embeddings=[embed(question)], n_results=3)
    context = "\n".join(hits["documents"][0])
    start = time.perf_counter()
    response = chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Context:\n{context}\nQuestion: {question}"}])
    return {"model": model, "question": question, "context": context, "answer": response, "seconds": round(time.perf_counter()-start, 3)}

def main():
    Path("outputs").mkdir(exist_ok=True)
    col = setup()
    rows = [answer(col, q) for q in QUESTIONS]
    rows += [answer(col, q, "qwen2.5:3b") for q in QUESTIONS[:3]]
    Path("outputs/results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open("outputs/comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for r in rows: print(f"{r['model']}: {r['question']} -> {r['answer']}")
    print("Review responses manually for correctness, relevance and hallucinations.")

if __name__ == "__main__": main()
