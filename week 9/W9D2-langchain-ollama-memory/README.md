# LangChain + Ollama Conversational Research Assistant

Create a LangChain prompt/model/output-parser chain powered by a local Ollama model, with explicit conversation history passed between turns.

## Setup

```bash
python -m venv .venv
# activate .venv
pip install -r requirements.txt
pip install -e .
```

## Run
Install Ollama and pull a model (default `llama3.1`), then:
```bash
pip install -r requirements.txt
ollama pull llama3.1
python examples/chat.py
```
Conversation history is held in the running Python process; it is not persistent across restarts.

## Repository contents

This standalone folder contains the shared report models, search adapter, CLI and documentation needed by the focused implementation. Optional integrations are imported only when used. See `docs/ARCHITECTURE.md` and `docs/VIVA.md` for design and review notes.

## Configuration and safety

Copy `.env.example` to `.env` only when needed. Never commit credentials. Offline mode is deterministic and does not claim live research.
