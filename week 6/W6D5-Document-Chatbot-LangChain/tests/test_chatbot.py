from src.chatbot import FACTS, calculator, document_search
def test_facts(): assert len(FACTS)==5
def test_calculator(): assert calculator.invoke({'expression':'12 * 3'}) == '36.0'
def test_search(): assert 'ChromaDB' in document_search.invoke({'query':'ChromaDB'})
