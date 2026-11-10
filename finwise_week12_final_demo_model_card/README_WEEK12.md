# FinWise – Week 12: Final Demo + Model Card

This package contains the model card, final serving code, trained model
artifact, Docker deployment configuration, GitHub Actions CI workflow,
integration tests, and a five-minute demo script.

## Model card
Read `MODEL_CARD.md` for task, data, metrics, intended use, limitations,
fairness/bias audit caveats, and deployment guidance.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000/docs for interactive API demonstration.

## Verify tests
```bash
pytest -q
```

## Docker
```bash
docker build -t finwise-credit-risk-api .
docker run --rm -p 8000:8000 finwise-credit-risk-api
```

## CI
`.github/workflows/ci.yml` contains the GitHub Actions workflow.
Push this folder to a GitHub repository and confirm the workflow passes.

## Demo
Follow `FIVE_MINUTE_DEMO_SCRIPT.txt`. It is a recording guide, not a
completed video. Record and upload the actual five-minute demonstration
separately.
