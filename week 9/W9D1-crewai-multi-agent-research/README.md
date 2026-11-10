# CrewAI Multi-Agent Research Crew

Build a three-agent research workflow: Research Analyst → Technical Report Writer → Quality Reviewer. It includes a web-search tool and a deterministic offline baseline.

## Setup

```bash
python -m venv .venv
# activate .venv
pip install -r requirements.txt
pip install -e .
```

## Run
```bash
python -m venv .venv
# activate the environment, then:
pip install -r requirements.txt
pip install -e .
python -m research_agent.cli --topic "AI in healthcare" --mode offline
python -m research_agent.cli --topic "AI in healthcare" --mode crew
```
Crew mode requires a supported LLM provider API key in `.env`.

## Repository contents

This standalone folder contains the shared report models, search adapter, CLI and documentation needed by the focused implementation. Optional integrations are imported only when used. See `docs/ARCHITECTURE.md` and `docs/VIVA.md` for design and review notes.

## Configuration and safety

Copy `.env.example` to `.env` only when needed. Never commit credentials. Offline mode is deterministic and does not claim live research.
