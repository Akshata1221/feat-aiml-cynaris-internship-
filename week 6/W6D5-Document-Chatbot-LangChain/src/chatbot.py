"""Document-grounded chatbot with conversational history and two safe tools."""
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent

FACTS=['RAG combines retrieval with language generation.','ChromaDB stores document embeddings.','Ollama supports running LLMs locally.','LangChain composes prompts, models, and parsers.','MLflow tracks machine learning experiments.']
@tool
def document_search(query: str) -> str:
    """Search the demonstration document for relevant terms."""
    terms=query.lower().split()
    matches=[fact for fact in FACTS if any(t.strip('?.!,') in fact.lower() for t in terms if len(t)>3)]
    return '\n'.join(matches) if matches else 'No matching facts.'
@tool
def calculator(expression: str) -> str:
    """Calculate simple two-number arithmetic like 12 * 3."""
    import re,operator
    match=re.fullmatch(r'\s*(-?\d+(?:\.\d+)?)\s*([+*/-])\s*(-?\d+(?:\.\d+)?)\s*',expression)
    if not match:return 'Unsupported expression'
    a,op,b=match.groups()
    try:return str({'+':operator.add,'-':operator.sub,'*':operator.mul,'/':operator.truediv}[op](float(a),float(b)))
    except ZeroDivisionError:return 'Cannot divide by zero'
def main():
    out=Path('outputs');out.mkdir(exist_ok=True)
    llm=ChatOllama(model='llama3.2:3b',temperature=0)
    store=Chroma.from_documents([Document(page_content=s) for s in FACTS],OllamaEmbeddings(model='nomic-embed-text'),collection_name='w6d5_facts')
    chain=ChatPromptTemplate.from_messages([('system','Use only context to answer. If absent say I do not know. Context: {context}'),('human','{question}')])|llm|StrOutputParser()
    history=[];logs=[]
    for question in ['What is RAG?','What stores embeddings?','What runs locally?','What does MLflow track?','Summarize the document.']:
        docs=store.similarity_search(question,k=3)
        answer=chain.invoke({'context':'\n'.join(d.page_content for d in docs),'question':question})
        history.extend([HumanMessage(content=question),AIMessage(content=answer)])
        logs.append(f'Q: {question}\nA: {answer}')
    # History is explicitly included in a follow-up prompt.
    followup=llm.invoke([('system','Answer using the preceding conversation.')]+history+[HumanMessage(content='What did I ask first?')])
    logs.append('FOLLOWUP: '+str(followup.content))
    agent=create_react_agent(llm,[document_search,calculator])
    for task in ['Use calculator for 12 * 3','Use document_search to find ChromaDB','Use document_search to find Ollama']:
        result=agent.invoke({'messages':[('user',task)]})
        logs.append('AGENT: '+str(result['messages'][-1].content))
    (out/'conversation.txt').write_text('\n\n'.join(logs),encoding='utf-8')
    print('\n\n'.join(logs))
if __name__=='__main__':main()
