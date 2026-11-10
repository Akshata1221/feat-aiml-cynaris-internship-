# Viva preparation

1. **Experiment vs run:** An experiment groups related runs and stores metadata; a run is one execution with its own parameters, metrics, tags, and artifacts.
2. **Registry vs artifact logging:** Artifact logging stores a model file in a run. The registry adds governed model versions, aliases, lineage, descriptions, and lifecycle management.
3. **A/B testing:** Register two model versions, route a controlled traffic split to each, log model version and cohort with prediction outcomes, and compare business and quality metrics before promoting a winner.
