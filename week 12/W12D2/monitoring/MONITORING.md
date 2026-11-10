# Production monitoring and retraining strategy

## Service and infrastructure signals
- Request rate, p50/p95/p99 latency, HTTP 4xx/5xx rates, timeout rate.
- Container CPU, memory, restart count, health-check failures and replica availability.
- Alert: page on-call when 5xx exceeds 2% for 5 minutes, p95 latency exceeds 500 ms for 10 minutes, or health checks fail twice consecutively.
- Alert: warn when CPU or memory exceeds 80% for 15 minutes; investigate repeated restarts immediately.

## Model and data quality
- Log model version, request ID, prediction class, feature schema version and latency.
- Do not log raw personal/sensitive input. Apply retention and access controls.
- Monitor feature missingness, invalid-value rate, feature distributions and prediction-class proportions.
- With delayed labels, track accuracy, precision/recall, F1 and calibration by segment.
- Use PSI or KS tests for numeric feature drift; alert when PSI > 0.2 (warning) and > 0.3 (critical), after validating thresholds against the domain.
- Alert on schema changes, unexpected categories, or missing features immediately.

## Retraining triggers and governance
Retraining is proposed when (a) labelled performance falls below the agreed SLO for two evaluation windows, (b) sustained feature drift is confirmed and supported by fresh representative labels, or (c) a scheduled review finds material data or business changes.
Drift alone is not an automatic production promotion signal. Create a candidate, compare against the current champion on a time-based holdout and important slices, run bias/safety checks, record metrics and artifacts in MLflow, obtain approval, then canary and monitor before full rollout. Keep rollback to the previous model version available.

## Operational process
1. Dashboards: API golden signals, container resources, feature drift, model quality, class distribution.
2. Alert routing: warning to team channel; critical to on-call with runbook link.
3. Incident response: acknowledge, assess blast radius, rollback or disable predictions if needed, preserve evidence, document root cause.
4. Review thresholds monthly; avoid noisy alerts and use minimum sample sizes.
