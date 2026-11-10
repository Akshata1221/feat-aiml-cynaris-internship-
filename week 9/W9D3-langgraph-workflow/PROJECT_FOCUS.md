# LangGraph Research Workflow

Represent the research pipeline as a compiled LangGraph state graph. The graph accepts a topic, runs the research/report node and returns Markdown.

## Run
```bash
pip install -r requirements.txt
python -m research_agent.cli --topic "Responsible AI" --mode graph
```
The search adapter falls back to a deterministic source set if live search is unavailable.
