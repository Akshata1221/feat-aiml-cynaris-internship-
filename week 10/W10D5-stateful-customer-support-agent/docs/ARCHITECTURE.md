# Architecture

Integrated support workflow with intent routing, checkpointed conversation memory, human escalation, optional MLflow tracking and Ragas evaluation scaffolding.

The implementation uses LangGraph `StateGraph`, typed state, explicit nodes, and graph edges. Conditional routing is used where multiple destinations are possible. Checkpoint-backed workflows must use a stable `thread_id`; interrupts pause execution and resume with `Command(resume=...)`.
