# Architecture

Pause before sending a response, collect approval or edits, and resume with Command(resume=...).

The implementation uses LangGraph `StateGraph`, typed state, explicit nodes, and graph edges. Conditional routing is used where multiple destinations are possible. Checkpoint-backed workflows must use a stable `thread_id`; interrupts pause execution and resume with `Command(resume=...)`.
