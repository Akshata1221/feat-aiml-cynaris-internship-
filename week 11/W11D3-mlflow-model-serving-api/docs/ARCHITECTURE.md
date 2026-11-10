# Architecture

Train and register a sklearn classifier, then serve the registered model through MLflow's REST API and call its predictions endpoint.

Keep data preparation, training/retrieval, tracking, and evaluation as separate functions so each part can be tested independently.
