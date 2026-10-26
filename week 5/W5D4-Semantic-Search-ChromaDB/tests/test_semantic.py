from src.semantic_search import DOCS, pdf_chunks
def test_twenty_docs(): assert len(DOCS) == 20
def test_metadata(): assert all("topic" in row[2] for row in DOCS)
def test_missing_pdf():
    import pytest
    with pytest.raises(FileNotFoundError): pdf_chunks("missing-file.pdf")
