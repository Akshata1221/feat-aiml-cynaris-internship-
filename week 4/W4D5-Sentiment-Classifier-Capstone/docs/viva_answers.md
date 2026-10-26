1. I built a sentiment classifier comparing TF-IDF Logistic Regression and Random Forest, evaluated both, saved the best, and exposed a FastAPI endpoint.
2. Preventing data leakage was the hardest part. I used a Pipeline so TF-IDF learns only from training text.
3. I would replace synthetic reviews with a large, real labeled dataset, tune hyperparameters, and deploy with CI/CD and monitoring.
