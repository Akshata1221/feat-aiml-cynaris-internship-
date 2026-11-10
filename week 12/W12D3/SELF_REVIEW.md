# Self-review checklist

- [ ] API validates payload shape and rejects invalid/non-finite values.
- [ ] Model artifact is reproducibly generated during image build.
- [ ] Container starts, serves `/health` and `/predict`, and runs as non-root.
- [ ] Docker image builds locally and health check passes.
- [ ] Ruff lint and pytest pass locally.
- [ ] GitHub Actions runs lint -> test -> build -> publish.
- [ ] GHCR package visibility and permissions are reviewed.
- [ ] Secrets are not committed; workflow uses scoped `GITHUB_TOKEN`.
- [ ] Monitoring thresholds and retraining governance are reviewed for the use case.
- [ ] Evidence screenshots are genuine and added.
- [ ] Two CIA Full Stack Mentor Mode reviews are completed and logged.
- [ ] At least two descriptive commits exist; PR is opened and reviewed.
