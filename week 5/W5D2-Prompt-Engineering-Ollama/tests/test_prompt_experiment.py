from unittest.mock import Mock, patch
from src.prompt_experiment import ask, FIVE_PROMPTS, COMPARISON, SYSTEM_PROMPTS

def test_prompt_counts():
    assert len(FIVE_PROMPTS) == 5 and len(COMPARISON) == 3

def test_system_styles():
    assert len(SYSTEM_PROMPTS) >= 3

@patch('src.prompt_experiment.requests.post')
def test_ollama_request(mock_post):
    mock_post.return_value = Mock(status_code=200)
    mock_post.return_value.json.return_value = {'message': {'content': 'hello'}}
    assert ask('llama3.2:3b', 'be concise', 'hi') == 'hello'
    payload=mock_post.call_args.kwargs['json']
    assert payload['messages'][0]['role']=='system'
    assert payload['stream'] is False
