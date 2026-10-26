import requests

BASE = "http://localhost:11434"

def chat(model, messages):
    r = requests.post(f"{BASE}/api/chat", json={"model": model, "messages": messages, "stream": False}, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]

def embed(text):
    r = requests.post(f"{BASE}/api/embed", json={"model": "nomic-embed-text", "input": text}, timeout=180)
    r.raise_for_status()
    return r.json()["embeddings"][0]
