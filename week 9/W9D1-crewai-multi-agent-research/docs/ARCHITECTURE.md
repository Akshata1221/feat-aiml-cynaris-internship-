# Architecture

```text
Topic
  |
  +--> Offline baseline / DuckDuckGo discovery
  |
  +--> CrewAI sequential crew
          Researcher --search tool--> Writer --> Reviewer
  |
  +--> Optional LangGraph state graph
  |
  +--> Markdown report --> MLflow artifact tracking
  |
  +--> Optional Ragas evaluation / LangChain-Ollama chat helper
```

## Agent responsibilities
- **Researcher:** gather relevant sources, capture URLs and evidence, flag uncertainty.
- **Writer:** synthesize research notes into a structured report without fabricating claims.
- **Reviewer:** inspect evidence, citations, limitations and clarity; return corrections.

CrewAI's sequential process passes task outputs through task context. Agents do not directly share arbitrary Python memory; the crew/task orchestration provides the context and artifacts needed by downstream tasks.
