# W12D1: MLOps overview — CI/CD foundations

Focus: understand the full CI/CD path and how containerisation, tests, publishing and monitoring fit together.

This is a self-contained, GitHub-ready implementation of the shared Week 12 practical deliverable.

## Included
- FastAPI ML prediction API with input validation, `/health`, `/ready`, `/predict`
- Reproducible scikit-learn model training script
- Dockerfile, `.dockerignore`, Compose file and container health check
- GitHub Actions: Ruff lint -> pytest -> Docker build and GHCR publish
- Monitoring, alerting and retraining strategy
- Architecture, local run/evidence instructions, self-review and viva preparation

## Quick start
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

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Docker
```bash
docker build -t production-ml-api:local .
docker run --rm -p 8000:8000 production-ml-api:local
```
Then test `http://localhost:8000/health`.

## CI/CD registry setup
The workflow publishes to `ghcr.io/<owner>/<repository>` on pushes (not pull requests).
Ensure GitHub Actions is enabled and the repository allows `GITHUB_TOKEN` package write
permissions. No long-lived registry password is required. Review package visibility after
the first successful publish.

## Git workflow
```bash
git switch -c feat/aiml-W12-akshata
git add app scripts tests requirements.txt requirements-dev.txt
git commit -m "feat: add validated ML prediction API and model"
git add .
git commit -m "feat: mlops — dockerised ml api with ci pipeline"
git push -u origin feat/aiml-W12-akshata
```
Open a pull request to `main`. Replace the branch suffix if your preferred GitHub handle differs.
The required headline commit can be used as an additional commit:
`feat: mlops — dockerised ml api with ci pipeline`

## Important validation notes
Source and workflow are prepared, but this environment has not executed Docker, GitHub Actions,
GHCR publishing, or a live deployment. Run the steps above and capture real outputs in `evidence/`.
The workflow's image publishing step is skipped on pull requests and runs on branch pushes.
