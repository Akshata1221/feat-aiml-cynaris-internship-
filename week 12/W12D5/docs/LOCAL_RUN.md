# Local run and evidence

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train.py
ruff check app scripts tests
pytest -q
uvicorn app.main:app --reload
```

In another terminal:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"features\":[5.1,3.5,1.4,0.2]}"
```

Docker:
```bash
docker build -t production-ml-api:local .
docker run --rm -p 8000:8000 production-ml-api:local
# Or: docker compose up --build
```

Capture screenshots of: successful test output, Docker build, running container, `/health`, `/predict`, and green GitHub Actions run. Store your own screenshots under `evidence/`; do not fabricate evidence.
