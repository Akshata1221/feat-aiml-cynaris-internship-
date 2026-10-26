import json
from pathlib import Path
from .chains import make_chain, make_memory_chain
from .agent import build_agent

def main():
    Path("outputs").mkdir(exist_ok=True)
    chain = make_chain()
    questions = ["Define an LLM", "What is a prompt template?", "Explain embeddings", "What is RAG?", "Explain an agent"]
    simple = [{"question": q, "answer": chain.invoke({"question": q})} for q in questions]
    memory, histories = make_memory_chain()
    turns = ["My favorite language is Python.", "I am learning LangChain.", "What language do I like?", "What am I learning?", "Summarize our chat."]
    chat = [{"question": q, "answer": memory.invoke({"question": q}, config={"configurable": {"session_id": "demo"}})} for q in turns]
    agent = build_agent()
    tasks = ["Use calculator to compute 12*(5+3)", "Use web_search_stub to search for AI news and tell me it is simulated", "Calculate 144/12 using calculator"]
    actions = []
    for task in tasks:
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
        actions.append({"task": task, "answer": str(result["messages"][-1].content)})
    data = {"chain": simple, "memory": chat, "history_message_count": len(histories["demo"].messages), "agent": actions}
    Path("outputs/results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))

if __name__ == "__main__": main()
