# LangGraph Research Workflow

Represent the research pipeline as a compiled LangGraph state graph. The graph accepts a topic, runs the research/report node and returns Markdown.

## Setup

```bash
python -m venv .venv
# activate .venv
pip install -r requirements.txt
pip install -e .
```

## Run
```bash
pip install -r requirements.txt
python -m research_agent.cli --topic "Responsible AI" --mode graph
```
The search adapter falls back to a deterministic source set if live search is unavailable.

## Repository contents

This standalone folder contains the shared report models, search adapter, CLI and documentation needed by the focused implementation. Optional integrations are imported only when used. See `docs/ARCHITECTURE.md` and `docs/VIVA.md` for design and review notes.

## Configuration and safety

Copy `.env.example` to `.env` only when needed. Never commit credentials. Offline mode is deterministic and does not claim live research.
