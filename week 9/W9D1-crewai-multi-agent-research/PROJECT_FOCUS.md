# CrewAI Multi-Agent Research Crew

Build a three-agent research workflow: Research Analyst → Technical Report Writer → Quality Reviewer. It includes a web-search tool and a deterministic offline baseline.

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
