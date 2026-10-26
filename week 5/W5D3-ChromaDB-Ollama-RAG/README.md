# W5D3 — ChromaDB Vector Store and Ollama PDF RAG

Creates a cosine-distance Chroma collection with **20 documents**, performs similarity and metadata-filtered searches, generates a small text-based sample PDF, extracts/chunks it, embeds chunks using local Ollama `nomic-embed-text`, retrieves the top three, and asks `llama3.2:3b` a grounded question. MLflow is optional. No fake evidence is bundled.

## Windows CMD setup
1. Install Ollama: https://ollama.com/download/windows
2. `ollama pull llama3.2:3b`
3. `ollama pull nomic-embed-text`
4. Ensure Ollama is running (`ollama serve` if needed).
5. `py -m venv .venv`
6. `.venv\Scripts\activate`
7. `pip install -r requirements.txt`
8. `python -m pytest -v`
9. `python -m src.rag_demo --mlflow`

Optional: `python -m src.rag_demo --pdf "data/my_document.pdf" --question "What is the document about?"`. Use a **text-based** PDF (scanned PDFs need OCR). Inspect `outputs/retrieval_report.json` and complete `docs/manual_verification.md`. This project uses a deterministic toy corpus, not a production-grade benchmark. The approved MLOps stack is represented by MLflow; CrewAI, LangGraph, and Ragas are not required for this foundational vector-store exercise.

## Git
Create a separate W5D3 GitHub repository. Before committing, obtain two real CIA Full Stack Mentor reviews and fill in `docs/cia_interactions.md`. Run `git init`, `git checkout -b feat/aiml-W5-Akshata`, `git add .`, `git commit -m "feat: chromadb — vector store setup and retrieval"`. After testing and documenting findings, `git add .` then `git commit -m "docs: verify PDF retrieval and add test evidence"`. Add your GitHub remote, `git push -u origin feat/aiml-W5-Akshata`, then raise a PR to main. Generated evidence is ignored by default; use `git add -f outputs/retrieval_report.json` only after reviewing it for private content.
