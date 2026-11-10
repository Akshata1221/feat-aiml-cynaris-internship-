# Viva preparation

1. An experiment groups related MLflow runs; each run records one training or evaluation execution.
2. Model Registry adds versioned governance and aliases, while artifact logging only stores a model output.
3. Use a gateway to split traffic between model aliases and log version/cohort plus prediction outcomes.
