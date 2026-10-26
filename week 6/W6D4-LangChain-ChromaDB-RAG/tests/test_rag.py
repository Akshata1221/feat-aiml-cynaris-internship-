from src.rag import sample_documents, TOPICS
def test_count(): assert len(sample_documents()) == 20
def test_metadata(): assert sample_documents()[0].metadata['group'] == 'software'
def test_topics(): assert len(set(TOPICS)) == 20
