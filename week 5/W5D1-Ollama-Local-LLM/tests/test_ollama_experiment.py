from unittest.mock import Mock, patch
from src.ollama_experiment import query_model, FIVE_PROMPTS, COMPARE_QUESTIONS


def test_five_prompts_and_three_comparison_questions():
    assert len(FIVE_PROMPTS) == 5
    assert len(COMPARE_QUESTIONS) == 3


def test_query_model_uses_system_prompt_and_non_streaming():
    fake = Mock()
    fake.json.return_value = {'message': {'content': 'Sample answer'}, 'total_duration': 123}
    with patch('src.ollama_experiment.requests.post', return_value=fake) as post:
        result = query_model('llama3.2:3b', 'Hello')
    assert result['response'] == 'Sample answer'
    assert result['total_duration_ns'] == 123
    assert post.call_args.kwargs['json']['messages'][0]['role'] == 'system'
    assert post.call_args.kwargs['json']['stream'] is False


def test_http_error_propagates():
    import pytest
    import requests
    fake = Mock()
    fake.raise_for_status.side_effect = requests.HTTPError('Bad request')
    with patch('src.ollama_experiment.requests.post', return_value=fake):
        with pytest.raises(requests.HTTPError):
            query_model('llama3.2:3b', 'Hello')
