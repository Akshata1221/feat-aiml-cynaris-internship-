from pathlib import Path

DEFAULT_DOCS = [
    "Refunds are available within 30 days when a receipt is provided.",
    "Standard support hours are Monday to Friday, 9 AM to 5 PM.",
    "Reset a password using the Forgot Password link on the sign-in page.",
    "A damaged item should be reported within seven days of delivery.",
]

def build_vector_store(documents=None, persist_directory="./chroma_db", chunk_size=500):
    """Optional local RAG setup. Install the rag extra before calling."""
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -e '.[rag']") from exc
    docs = [Document(page_content=x) for x in (documents or DEFAULT_DOCS)]
    chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50).split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

def retrieve_and_answer(question, vector_store, k=3):
    docs = vector_store.similarity_search(question, k=k)
    context = "\n".join(doc.page_content for doc in docs)
    # Deterministic baseline answer; replace with your approved chat model.
    answer = context.split("\n")[0] if context else "I could not find relevant information."
    return {"question": question, "answer": answer, "contexts": [d.page_content for d in docs]}
