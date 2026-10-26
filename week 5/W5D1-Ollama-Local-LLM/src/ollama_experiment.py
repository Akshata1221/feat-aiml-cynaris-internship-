"""Run local Ollama inference and compare two models on identical prompts."""
import argparse
import json
import os
import time
from pathlib import Path
import requests

API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/chat')
SYSTEM_PROMPT = 'You are a helpful AI tutor. Answer accurately, clearly, and concisely.'
FIVE_PROMPTS = [
    'Explain machine learning in two sentences.',
    'Write a Python function to check whether a number is prime.',
    'What is the difference between precision and recall?',
    'Summarize the advantages of running LLMs locally.',
    'Give three ways to reduce overfitting in machine learning.',
]
COMPARE_QUESTIONS = FIVE_PROMPTS[:3]
MODELS = ('llama3.2:3b', 'qwen2.5:3b')


def query_model(model, prompt, system_prompt=SYSTEM_PROMPT, timeout=180):
    """Return response and duration; raise on API/HTTP failure."""
    start = time.perf_counter()
    response = requests.post(API_URL, json={
        'model': model,
        'messages': [{'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': prompt}],
        'stream': False,
        'options': {'temperature': 0},
    }, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    answer = payload['message']['content']
    return {'model': model, 'prompt': prompt, 'response': answer,
            'elapsed_seconds': round(time.perf_counter() - start, 3),
            'total_duration_ns': payload.get('total_duration')}


def run_experiment(output_dir=Path('outputs'), track_mlflow=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    # Five initial prompts on llama; three shared prompts on qwen.
    for model, prompts in [(MODELS[0], FIVE_PROMPTS), (MODELS[1], COMPARE_QUESTIONS)]:
        for prompt in prompts:
            print(f'Running {model}: {prompt}')
            result = query_model(model, prompt)
            results.append(result)
            print(result['response'][:250], '\n')
    (output_dir / 'responses.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    import csv
    with (output_dir / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['model', 'prompt', 'elapsed_seconds', 'response'])
        writer.writeheader()
        writer.writerows({key: result[key] for key in writer.fieldnames} for result in results)
    if track_mlflow:
        import mlflow
        mlflow.set_experiment('W5D1_Local_Ollama_Comparison')
        for result in results:
            with mlflow.start_run(run_name=result['model'].replace(':', '_')):
                mlflow.log_param('model', result['model'])
                mlflow.log_param('prompt', result['prompt'])
                mlflow.log_param('temperature', 0)
                mlflow.log_metric('elapsed_seconds', result['elapsed_seconds'])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow', action='store_true', help='Track latency in local MLflow')
    args = parser.parse_args()
    try:
        run_experiment(track_mlflow=args.mlflow)
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit('Cannot connect to Ollama. Start Ollama and verify http://localhost:11434') from exc
    except requests.exceptions.HTTPError as exc:
        raise SystemExit('Ollama API error. Check that both models have been pulled.') from exc
