"""Optional LangChain + Ollama chain with per-session conversation history."""
def build_ollama_chain():
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as exc:
        raise RuntimeError("Install langchain and langchain-ollama to use this feature.") from exc
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful research assistant. Be concise and flag uncertainty."),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])
    model = ChatOllama(model=__import__("os").getenv("OLLAMA_MODEL", "llama3.1"))
    return prompt | model | StrOutputParser()

def answer_with_history(question: str, history: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    chain = build_ollama_chain()
    messages = []
    for human, ai in history:
        messages.extend([("human", human), ("ai", ai)])
    answer = chain.invoke({"question": question, "history": messages})
    return answer, history + [(question, answer)]
