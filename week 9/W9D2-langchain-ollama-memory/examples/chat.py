from research_agent.hybrid import answer_with_history

history = []
print("Research assistant ready. Type exit to quit.")
while True:
    question = input("You: ").strip()
    if question.lower() in {"exit", "quit"}:
        break
    answer, history = answer_with_history(question, history)
    print(f"Assistant: {answer}\n")
