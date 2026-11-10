# Architecture

Thread-scoped conversation state, checkpointing, and a SQLite persistence option.

The implementation uses LangGraph `StateGraph`, typed state, explicit nodes, and graph edges. Conditional routing is used where multiple destinations are possible. Checkpoint-backed workflows must use a stable `thread_id`; interrupts pause execution and resume with `Command(resume=...)`.
