# Viva preparation

## 1. Sequential vs. hierarchical CrewAI process
In a sequential process, tasks execute in a defined order and later tasks can consume earlier task outputs. In a hierarchical process, a manager agent delegates work, assigns tasks and reviews results dynamically. Sequential is predictable for a fixed research pipeline; hierarchical is useful when delegation and task allocation are complex.

## 2. How does CrewAI handle agent communication?
CrewAI coordinates agents through the crew process and task context. Outputs from completed tasks can be supplied as context to later tasks. Agents can also use tools and delegation when configured. This project uses explicit task context in a sequential flow rather than relying on unstructured direct messaging.

## 3. When use CrewAI over a single LangChain agent?
Use CrewAI when the workflow benefits from multiple specialized roles, explicit task ownership and review stages. A single LangChain agent is often simpler for a narrow tool-using task where role separation adds overhead.

## 4. What was built and why?
A research report workflow with researcher, writer and reviewer responsibilities, a search tool, reproducible offline mode, optional graph orchestration and experiment tracking. The separation improves traceability and makes quality checks explicit.

## 5. Hardest part and solution
Live search and LLM services can fail due to network limits or missing credentials. The project separates search from report generation and provides a clearly labelled deterministic fallback, while requiring credentials for actual CrewAI execution.

## 6. What would be improved with one more day?
Add source credibility scoring, publication-date extraction, deduplication, a proper Ragas golden evaluation dataset, CI workflows and a UI for reviewing citations.
