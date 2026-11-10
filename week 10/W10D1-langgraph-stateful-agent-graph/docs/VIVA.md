# Viva preparation

1. **What did you build?** Three-node classify → route → respond graph, conditional routing, and human approval interrupts. The graph state is explicit, nodes perform small operations, and edges determine the next step.
2. **Hardest part?** Managing graph state and ensuring conditional transitions and interrupt resume use the same thread/checkpoint context.
3. **What would you improve?** Add typed input validation, richer observability, integration tests, production persistence, authentication, and robust failure/retry policies.
