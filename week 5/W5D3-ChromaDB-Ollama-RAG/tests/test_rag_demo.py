from src.rag_demo import DOCS, pdf_chunks, ensure_sample_pdf

def test_twenty_documents():
    assert len(DOCS) == 20
    assert all(category and text for category,text in DOCS)

def test_sample_pdf_extraction(tmp_path):
    path=tmp_path/'sample.pdf'
    ensure_sample_pdf(path)
    chunks=pdf_chunks(path)
    assert len(chunks)>0
    assert any('Precision' in text for _,text in chunks)

def test_pdf_chunk_size(tmp_path):
    path=tmp_path/'sample.pdf';ensure_sample_pdf(path)
    assert all(len(text)<=350 for _,text in pdf_chunks(path))
