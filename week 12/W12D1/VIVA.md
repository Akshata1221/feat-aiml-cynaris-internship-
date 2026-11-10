# Viva preparation

## 1. What did you build and why?
I built a FastAPI classification service around a persisted scikit-learn Iris model. Docker makes runtime dependencies reproducible. The container runs as a non-root user and includes a health check. GitHub Actions runs lint and tests before building and publishing an image to GHCR, so broken code is blocked before release. Monitoring covers service reliability, data drift and delayed-label model quality.

## 2. Hardest part and solution
The key challenge is making model artifacts and runtime environments reproducible. The Docker build trains and stores a deterministic model artifact inside the image, while API tests validate readiness, successful prediction and invalid inputs. CI repeats linting and tests in a clean environment.

## 3. What would you improve with one more day?
Add model/data versioning with MLflow, structured metrics and traces, a Prometheus/Grafana dashboard, automated drift evaluation, security scanning, image signing, deployment to a staging environment, and a canary/rollback workflow.
