"""Local Ollama embeddings + persistent Chroma vector retrieval for a sample PDF."""
import argparse
import json
from pathlib import Path
import requests

OLLAMA='http://127.0.0.1:11434'
DOCS = [
    ('ml', 'Supervised learning uses labelled examples.'),
    ('ml', 'Unsupervised learning discovers structure without labels.'),
    ('ml', 'Classification predicts categories.'),
    ('ml', 'Regression predicts continuous values.'),
    ('ml', 'Precision is TP divided by TP plus FP.'),
    ('ml', 'Recall is TP divided by TP plus FN.'),
    ('ml', 'F1 is the harmonic mean of precision and recall.'),
    ('ml', 'Overfitting means strong training performance but poor generalisation.'),
    ('ml', 'Cross-validation estimates performance across multiple splits.'),
    ('ml', 'Regularisation can limit model complexity.'),
    ('llm', 'An LLM predicts tokens conditioned on context.'),
    ('llm', 'A system prompt sets high-level behaviour instructions.'),
    ('llm', 'Temperature controls randomness in token sampling.'),
    ('llm', 'Quantisation reduces weight precision and memory use.'),
    ('llm', 'Embeddings represent text as numerical vectors.'),
    ('llm', 'Cosine similarity measures vector alignment.'),
    ('llm', 'ChromaDB stores and searches embedding vectors.'),
    ('llm', 'RAG retrieves external context before generation.'),
    ('llm', 'Chunking divides long documents into smaller segments.'),
    ('llm', 'Metadata filtering narrows a vector search to matching documents.'),
]

def embed(text, model='nomic-embed-text'):
    response=requests.post(f'{OLLAMA}/api/embed', json={'model':model,'input':text}, timeout=180)
    response.raise_for_status()
    return response.json()['embeddings'][0]

def chat(question, context, model='llama3.2:3b'):
    response=requests.post(f'{OLLAMA}/api/chat', json={'model':model,'stream':False,'messages':[{'role':'system','content':'Answer only from the supplied context. If absent, say you do not know. Cite chunk IDs when possible.'},{'role':'user','content':f'Context:\n{context}\n\nQuestion: {question}'}]}, timeout=180)
    response.raise_for_status()
    return response.json()['message']['content']

def ensure_sample_pdf(path):
    if path.exists(): return
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    path.parent.mkdir(parents=True, exist_ok=True)
    story=[]; styles=getSampleStyleSheet()
    for category, text in DOCS:
        story.extend([Paragraph(f'{category.upper()}: {text}',styles['BodyText']),Spacer(1,10)])
    SimpleDocTemplate(str(path)).build(story)

def pdf_chunks(path, max_chars=350):
    from pypdf import PdfReader
    pages=[page.extract_text() or '' for page in PdfReader(str(path)).pages]
    chunks=[]
    for page_index, page in enumerate(pages):
        lines=[line.strip() for line in page.splitlines() if line.strip()]
        chunk=''
        for line in lines:
            if chunk and len(chunk)+len(line)+1>max_chars:
                chunks.append((page_index+1,chunk));chunk=''
            chunk=(chunk+' '+line).strip()
        if chunk: chunks.append((page_index+1,chunk))
    return chunks

def main(pdf_path, question, model='llama3.2:3b', use_mlflow=False):
    import chromadb
    path=Path(pdf_path);ensure_sample_pdf(path)
    client=chromadb.PersistentClient(path='chroma_store')
    collection=client.get_or_create_collection('w5d3_docs',metadata={'hnsw:space':'cosine'})
    texts=[text for _,text in DOCS]
    collection.upsert(ids=[f'doc-{i}' for i in range(len(texts))],documents=texts,embeddings=[embed(t) for t in texts],metadatas=[{'category':category} for category,_ in DOCS])
    query_vector=embed('What is precision?')
    similar=collection.query(query_embeddings=[query_vector],n_results=3)
    filtered=collection.query(query_embeddings=[query_vector],n_results=3,where={'category':'ml'})
    chunks=pdf_chunks(path)
    if not chunks: raise ValueError('No extractable text in PDF; supply a text-based PDF.')
    pdf_collection=client.get_or_create_collection('w5d3_pdf',metadata={'hnsw:space':'cosine'})
    pdf_collection.upsert(ids=[f'chunk-{i}' for i in range(len(chunks))],documents=[t for _,t in chunks],embeddings=[embed(t) for _,t in chunks],metadatas=[{'page':p,'source':path.name} for p,_ in chunks])
    results=pdf_collection.query(query_embeddings=[embed(question)],n_results=min(3,len(chunks)),include=['documents','distances','metadatas'])
    context='\n'.join(f'[chunk {i}] {text}' for i,text in zip(results['ids'][0],results['documents'][0]))
    answer=chat(question,context,model)
    output=Path('outputs');output.mkdir(exist_ok=True)
    report={'question':question,'answer':answer,'retrieved':results,'similarity_search':similar,'metadata_filtered_search':filtered,'manual_verification':'PENDING: inspect retrieved text and answer'}
    (output/'retrieval_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Retrieved chunks:\n',context,'\nAnswer:\n',answer)
    if use_mlflow:
        import mlflow
        mlflow.set_experiment('W5D3_Chroma_Ollama_RAG')
        with mlflow.start_run():
            mlflow.log_param('llm',model);mlflow.log_param('embedding_model','nomic-embed-text');mlflow.log_metric('document_count',len(DOCS));mlflow.log_metric('pdf_chunk_count',len(chunks));mlflow.log_artifact(str(output/'retrieval_report.json'))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pdf',default='data/sample_ml_notes.pdf');parser.add_argument('--question',default='What is precision in machine learning?');parser.add_argument('--mlflow',action='store_true');args=parser.parse_args()
    main(args.pdf,args.question,use_mlflow=args.mlflow)
