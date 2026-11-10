from rag_evaluation.dataset import QUESTIONS, GROUND_TRUTHS

def test_ten_question_answer_pairs():
    assert len(QUESTIONS) == 10
    assert len(GROUND_TRUTHS) == 10
    assert all(q.strip() for q in QUESTIONS)
