from src.demo import calculate, web_search_stub
import pytest
def test_calculator(): assert calculate('2 + 3 * 4') == 14
def test_reject_code():
    with pytest.raises(ValueError): calculate('__import__("os").system("echo bad")')
def test_stub(): assert 'MOCK SEARCH' in web_search_stub.invoke({'query':'test'})
