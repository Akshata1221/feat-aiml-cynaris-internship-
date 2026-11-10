# FinWise – Week 9: Integration + Testing

This package contains the FastAPI serving layer for the Week 6 Random Forest model,
integration tests, and a repeatable load-test script.

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

API endpoints:
- GET /health
- POST /predict
- Swagger UI: /docs

## Integration tests
```bash
pytest -q
```

## Load test
Keep the API running in another terminal:
```bash
python scripts/load_test.py --requests 500 --concurrency 50
```

The included `load_test_results.json` records the observed run: 500 successful
requests, 50 concurrent workers, and approximately 32.54 requests/second.
Latency and throughput are environment-dependent.

## Docker
```bash
docker build -t finwise-credit-risk-api .
docker run --rm -p 8000:8000 finwise-credit-risk-api
```
