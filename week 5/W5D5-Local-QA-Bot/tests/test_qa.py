from src.local_qa import QUESTIONS, FACTS, SYSTEM
def test_five_questions(): assert len(QUESTIONS) == 5
def test_retrieval_corpus(): assert len(FACTS) >= 5
def test_grounded_prompt(): assert "context" in SYSTEM.lower()
