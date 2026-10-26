from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

MODEL = "llama3.2:3b"

def make_chain():
    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful concise tutor."), ("human", "{question}")])
    return prompt | ChatOllama(model=MODEL, temperature=0) | StrOutputParser()

def make_memory_chain():
    prompt = ChatPromptTemplate.from_messages([("system", "Remember the conversation and answer concisely."), MessagesPlaceholder("history"), ("human", "{question}")])
    base = prompt | ChatOllama(model=MODEL, temperature=0) | StrOutputParser()
    histories = {}
    def get_history(session_id):
        return histories.setdefault(session_id, InMemoryChatMessageHistory())
    return RunnableWithMessageHistory(base, get_history, input_messages_key="question", history_messages_key="history"), histories
