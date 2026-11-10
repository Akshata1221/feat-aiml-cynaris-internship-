# Model Card — FinWise Loan Default Risk Model

## Model details
- Model: RandomForestClassifier in a scikit-learn preprocessing pipeline
- Selected run: `rf_baseline`
- Intended task: estimate the probability of `default_flag=1` for a loan application
- Model artifact: `artifacts/best_model.joblib`
- Selection criterion: ROC-AUC on a stratified held-out test set

## Data
- Dataset: `finwise_loan_applications.csv`
- Rows: 1,000
- Inputs: age, annual income, employment type, employment years, existing loans,
  existing EMI, credit score, loan amount, loan purpose, LTV ratio and DTI ratio
- Target: `default_flag`
- Excluded identifier: `application_id`
- Split: 80% train / 20% test, stratified, random_state=42
- Data provenance and representativeness have not been independently verified.

## Performance
Held-out test results from the Week 6 experiment:
- Accuracy: 0.8650
- Precision: 0.8000
- Recall: 0.5833
- F1: 0.6747
- ROC-AUC: 0.9205
- PR-AUC: 0.8060

## Intended use and prohibited use
This is an educational prototype for demonstrating model training, serving,
experimentation and monitoring. It is not validated for real lending decisions,
credit eligibility, pricing, collections or other consequential decisions. Do not
use the output as the sole or decisive basis for an individual decision.

## Limitations and risks
- One random holdout split is not a substitute for temporal or external validation.
- Dataset provenance, collection process, label quality and representativeness need review.
- A fixed 0.5 classification threshold is illustrative and not cost-calibrated.
- Probability calibration has not been established.
- Input distribution drift and changing default patterns may reduce performance.
- Small subgroup counts make subgroup metrics unstable.

## Bias and fairness review
The supplied dataset does not contain an explicit protected-group label suitable for
an authorized fairness audit. A descriptive check by `employment_type` on the 200-row
test set produced the following results. Employment type is not a protected-class
substitute and these figures do not establish fairness or absence of bias.

| Employment type | Test rows | Observed default rate | Predicted positive rate | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| contract | 46 | 17.4% | 10.9% | 50.0% | 80.0% |
| government | 59 | 32.2% | 30.5% | 73.7% | 77.8% |
| salaried | 42 | 28.6% | 16.7% | 50.0% | 85.7% |
| self_employed | 53 | 17.0% | 9.4% | 44.4% | 80.0% |

A proper audit requires legally and ethically approved protected-group data, adequate
sample sizes, subgroup error-rate and calibration analysis, confidence intervals,
intersectional analysis, domain review and remediation. Do not infer protected traits.

## Deployment
Install `requirements.txt`, then run:
`uvicorn app.main:app --host 0.0.0.0 --port 8000`
Health: `GET /health`; prediction: `POST /predict`; interactive docs: `/docs`.
Docker instructions are in `README.md`. CI runs the pytest integration suite.

## Monitoring and maintenance
Monitor request volume, error rate, latency percentiles, feature schema, missingness,
input drift, prediction distribution, delayed outcome performance and subgroup
performance where lawful. Set alert thresholds with the deployment owner. Retrain only
after data-quality, validation, fairness and approval checks. Version model artifacts
and retain rollback capability.

## Human oversight
A qualified credit-risk and compliance team must validate data rights, legal basis,
model performance, explainability, fairness, threshold policy and adverse-action
requirements before any real-world use.
