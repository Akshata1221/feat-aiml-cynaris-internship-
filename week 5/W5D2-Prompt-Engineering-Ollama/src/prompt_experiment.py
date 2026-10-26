"""Reproducible Ollama prompt engineering experiment."""
import argparse
import csv
import json
import time
from pathlib import Path
import requests

API = "http://127.0.0.1:11434/api/chat"
SYSTEM_PROMPTS = {
    "concise": "You are a careful tutor. Respond in 2 short sentences; acknowledge uncertainty.",
    "structured": "You are a careful tutor. Respond using 3 numbered points; avoid invented facts.",
    "few_shot": "You are a sentiment labeler. Return exactly POSITIVE, NEGATIVE, or NEUTRAL. Examples: 'I love it' -> POSITIVE; 'It broke' -> NEGATIVE; 'It is a pen' -> NEUTRAL.",
}
FIVE_PROMPTS = ["Explain overfitting to a beginner.", "What is precision in ML?", "Explain a vector database.", "Give two ways to reduce hallucinations.", "What does temperature control in text generation?"]
COMPARISON = ["Explain overfitting to a beginner.", "What is precision in ML?", "Explain a vector database."]

def ask(model, system, prompt, temperature=0.2, timeout=180):
    response = requests.post(API, json={"model": model, "stream": False, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "options": {"temperature": temperature, "seed": 42}}, timeout=timeout)
    response.raise_for_status()
    return response.json()["message"]["content"]

def run(models, use_mlflow=False):
    output = Path('outputs'); output.mkdir(exist_ok=True)
    rows = []
    for model in models:
        for style, system in SYSTEM_PROMPTS.items():
            prompts = FIVE_PROMPTS if model == models[0] else COMPARISON
            for prompt in prompts:
                start = time.perf_counter()
                try:
                    answer = ask(model, system, prompt)
                    error = ""
                except (requests.RequestException, KeyError, ValueError) as exc:
                    answer, error = "", str(exc)
                rows.append({"model": model, "style": style, "prompt": prompt, "response": answer, "seconds": round(time.perf_counter()-start, 2), "error": error})
                print(f'{model} | {style} | {prompt[:30]} | {"OK" if not error else error}')
    (output/'responses.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer=csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if use_mlflow:
        import mlflow
        mlflow.set_experiment('W5D2_Prompt_Engineering')
        with mlflow.start_run():
            mlflow.log_param('models', ','.join(models)); mlflow.log_param('temperature', 0.2)
            mlflow.log_metric('successful_calls', sum(not r['error'] for r in rows))
            mlflow.log_artifact(str(output/'comparison.csv'))
    return rows

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mlflow', action='store_true'); args=parser.parse_args()
    run(['llama3.2:3b', 'qwen2.5:3b'], args.mlflow)
