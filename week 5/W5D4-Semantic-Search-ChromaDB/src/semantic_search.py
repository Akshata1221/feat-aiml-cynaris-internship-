import json
from pathlib import Path
import chromadb
from pypdf import PdfReader
from .ollama_client import embed, chat

DOCS = [(f"doc-{i:02d}", f"{topic} is useful for {context}. Example number {i}.", {"topic": topic}) for i, (topic, context) in enumerate(
    [("python", "data processing"), ("ml", "classification"), ("vector", "semantic search"), ("api", "deployment"), ("database", "storage")] * 4)]

def collection(path="chroma_store"):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection("learning_docs", metadata={"hnsw:space": "cosine"})

def index_documents(col, docs=DOCS):
    col.upsert(ids=[x[0] for x in docs], documents=[x[1] for x in docs], metadatas=[x[2] for x in docs], embeddings=[embed(x[1]) for x in docs])

def search(col, query, topic=None, n=3):
    kwargs = {"query_embeddings": [embed(query)], "n_results": min(n, col.count())}
    if topic: kwargs["where"] = {"topic": topic}
    return col.query(**kwargs)

def pdf_chunks(path, chunk_size=650, overlap=100):
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if not text.strip(): raise ValueError("PDF has no extractable text")
    step = chunk_size - overlap
    return [text[i:i+chunk_size] for i in range(0, len(text), step) if text[i:i+chunk_size].strip()]

def pdf_qa(path, question, store="chroma_store"):
    col = chromadb.PersistentClient(path=store).get_or_create_collection("pdf_chunks", metadata={"hnsw:space": "cosine"})
    chunks = pdf_chunks(path)
    col.upsert(ids=[f"chunk-{i}" for i in range(len(chunks))], documents=chunks, embeddings=[embed(c) for c in chunks])
    hits = col.query(query_embeddings=[embed(question)], n_results=min(3, len(chunks)))
    context = "\n\n".join(hits["documents"][0])
    answer = chat("llama3.2:3b", [{"role": "system", "content": "Answer only using provided context. Say 'Not found' if absent."}, {"role": "user", "content": f"Context:\n{context}\nQuestion: {question}"}])
    return {"question": question, "chunks": hits["documents"][0], "answer": answer}

def main():
    Path("outputs").mkdir(exist_ok=True)
    col = collection(); index_documents(col)
    result = {"indexed": col.count(), "similarity": search(col, "semantic vector retrieval"), "filtered": search(col, "python data", "python")}
    from reportlab.pdfgen import canvas
    pdf = Path("outputs/sample.pdf")
    c = canvas.Canvas(str(pdf)); c.drawString(50, 750, "ChromaDB stores document embeddings for semantic search."); c.drawString(50, 725, "Ollama provides local embeddings and question answering."); c.save()
    result["pdf_qa"] = pdf_qa(pdf, "What does ChromaDB store?")
    Path("outputs/results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
