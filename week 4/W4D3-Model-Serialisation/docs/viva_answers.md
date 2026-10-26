# Viva answers

1. I trained a scaled logistic regression classifier, evaluated its precision, recall, F1 and AUC, and saved it using both joblib and pickle. I verified that reloaded models reproduce the original predictions.
2. The hardest part was ensuring that preprocessing was saved together with the estimator. I used a scikit-learn Pipeline so inference uses the same scaler.
3. With another day, I would add versioned artifacts, environment locking, a prediction API, and deployment tests.

**Security:** Only load joblib/pickle artifacts from trusted sources; both formats can execute malicious code during loading.
