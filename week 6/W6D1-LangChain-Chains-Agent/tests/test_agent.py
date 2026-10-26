import pytest
from src.agent import safe_calculate, web_search_stub
def test_calculator(): assert safe_calculate("12*(5+3)") == 96
def test_invalid_expression():
    with pytest.raises(ValueError): safe_calculate("__import__('os')")
def test_stub_is_labeled(): assert "STUB ONLY" in web_search_stub.invoke({"query": "news"})
