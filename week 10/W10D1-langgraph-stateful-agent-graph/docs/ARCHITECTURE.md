# Architecture

Three-node classify → route → respond graph, conditional routing, and human approval interrupts.

The implementation uses LangGraph `StateGraph`, typed state, explicit nodes, and graph edges. Conditional routing is used where multiple destinations are possible. Checkpoint-backed workflows must use a stable `thread_id`; interrupts pause execution and resume with `Command(resume=...)`.
