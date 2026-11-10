# Architecture

Explicit state-machine transitions with validated conditional routing and five-input tests.

The implementation uses LangGraph `StateGraph`, typed state, explicit nodes, and graph edges. Conditional routing is used where multiple destinations are possible. Checkpoint-backed workflows must use a stable `thread_id`; interrupts pause execution and resume with `Command(resume=...)`.
