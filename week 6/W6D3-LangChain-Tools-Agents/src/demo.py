"""LangChain chain, five-turn history, and two-tool LangGraph agent."""
import ast
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
def calculate(expression: str):
    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int,float): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS: return OPS[type(node.op)](visit(node.left),visit(node.right))
        raise ValueError('Unsupported expression')
    return visit(ast.parse(expression, mode='eval'))

@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression like 12 * 4."""
    return str(calculate(expression))

@tool
def web_search_stub(query: str) -> str:
    """Return mock search data for demonstration; this does not access the internet."""
    return f'MOCK SEARCH RESULT for {query}: This is a demonstration, not a real web result.'

def main():
    llm = ChatOllama(model='llama3.2:3b', temperature=0)
    chain = ChatPromptTemplate.from_messages([('system','Answer clearly in one sentence.'),('human','{question}')]) | llm | StrOutputParser()
    for question in ['What is Python?','What is an embedding?','Define RAG.','What is a vector database?','What is an agent?']:
        print('CHAIN',question,chain.invoke({'question':question}))
    history=[]
    for message in ['My name is Sam.','I study Python.','What do I study?','What is my name?','Summarize what you remember about me.']:
        reply=llm.invoke([('system','Use conversation history to answer accurately.')]+history+[HumanMessage(content=message)])
        history.extend([HumanMessage(content=message),AIMessage(content=reply.content)])
        print('MEMORY',message,reply.content)
    agent=create_react_agent(llm,[calculator,web_search_stub])
    for task in ['What is 13 * 7?','Use web_search_stub to search for LangChain.','What is 100 / 4?']:
        result=agent.invoke({'messages':[('user',task)]})
        print('AGENT',task,result['messages'][-1].content)
if __name__=='__main__': main()
