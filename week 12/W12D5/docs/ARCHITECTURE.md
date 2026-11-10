# Architecture

Client -> FastAPI `/predict` -> validated four-feature payload -> persisted scikit-learn model -> class prediction.
`/health` and `/ready` expose readiness. Docker builds a self-contained image and runs as a non-root user. GitHub Actions runs Ruff and pytest before building and publishing an image to GitHub Container Registry (GHCR). Monitoring guidance covers service health, data drift, model quality and controlled retraining.
