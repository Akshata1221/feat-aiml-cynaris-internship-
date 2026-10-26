# W5D4-Semantic-Search-ChromaDB

Cynaris AI/ML practical project. Requires local Ollama running. Real model outputs are generated on your machine, not bundled.

## Setup (Windows CMD, from extracted project folder)
```bat
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v
python -m src.semantic_search
```
If Ollama is not already running, launch the Ollama desktop application or run `ollama serve` in another terminal. Results are written to `outputs/`.

## Submission
Branch: `feat/aiml-W5-Akshata`. Obtain two genuine CIA Full Stack Mentor reviews **before** committing; record feedback in `docs/cia_interactions.md`. Make at least two descriptive commits, push and open a PR. Save genuine screenshots of outputs and tests.

## Limitations
Ollama requires downloaded models and sufficient RAM. Tests are offline unit tests; full integration must be verified locally. The web-search tool is deliberately a stub, not real search. Do not claim the outputs are independently verified until you inspect them.
