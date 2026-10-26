"""Persistent Chroma semantic search and PDF-grounded Ollama RAG."""
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

TOPICS=['Python','SQL','Machine Learning','Deep Learning','NLP','Computer Vision','RAG','Embeddings','ChromaDB','Ollama','LangChain','LangGraph','CrewAI','MLflow','Ragas','MLOps','Classification','Regression','Precision','Recall']
def sample_documents():
    return [Document(page_content=f'{topic} is an important topic in applied AI and software engineering. Document number {i}.',metadata={'topic':topic,'group':'ai' if i>2 else 'software'}) for i,topic in enumerate(TOPICS)]
def make_pdf(path):
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=['RAG retrieves relevant document passages before generating answers.', 'ChromaDB stores vectors and metadata for semantic retrieval.', 'Ollama runs language and embedding models locally.']
    SimpleDocTemplate(str(path)).build([Paragraph(s,getSampleStyleSheet()['Normal']) for s in lines])
def main():
    from langchain_community.document_loaders import PyPDFLoader
    out=Path('outputs');out.mkdir(exist_ok=True)
    embeddings=OllamaEmbeddings(model='nomic-embed-text')
    store=Chroma(collection_name='w6d4_documents',persist_directory=str(out/'chroma'),embedding_function=embeddings,collection_metadata={'hnsw:space':'cosine'})
    docs=sample_documents()
    store.add_documents(docs,ids=[f'doc-{i}' for i in range(len(docs))])
    for doc,dist in store.similarity_search_with_score('What is retrieval augmented generation?',k=3):print('SEARCH',dist,doc.page_content)
    print('FILTER',[(d.metadata) for d in store.similarity_search('programming',k=3,filter={'group':'software'})])
    pdf=out/'sample.pdf';make_pdf(pdf)
    chunks=RecursiveCharacterTextSplitter(chunk_size=180,chunk_overlap=30).split_documents(PyPDFLoader(str(pdf)).load())
    pdf_store=Chroma.from_documents(chunks,embeddings,collection_name='w6d4_pdf',persist_directory=str(out/'pdf_chroma'))
    question='What does RAG do?'
    retrieved=pdf_store.similarity_search(question,k=3)
    context='\n'.join(d.page_content for d in retrieved)
    chain=ChatPromptTemplate.from_messages([('system','Answer only from this context. If missing, say I do not know. Context: {context}'),('human','{question}')]) | ChatOllama(model='llama3.2:3b',temperature=0) | StrOutputParser()
    answer=chain.invoke({'context':context,'question':question})
    (out/'answer.txt').write_text(f'Question: {question}\nContext: {context}\nAnswer: {answer}',encoding='utf-8')
    print('ANSWER',answer)
if __name__=='__main__':main()
