# Viva — suggested answers

**What did you build today and why?**
I built a Python project comparing GridSearchCV and RandomizedSearchCV for SVM and KNN. I used scaling inside pipelines, stratified cross-validation, and F1 scoring to tune fairly and avoid data leakage. MLflow records experiments and the output files show results.

**What was hardest and how did you solve it?**
Avoiding unfair comparisons and data leakage was the main challenge. I placed StandardScaler inside each pipeline and used the same data split and cross-validation strategy for all searches.

**What would you improve with one more day?**
I would add more classifiers, nested cross-validation, runtime benchmarks on larger search spaces, and automated CI tests.

**GridSearch vs RandomSearch?**
GridSearch checks every listed combination; RandomSearch samples a fixed number of combinations and can save computation. Both use cross-validation.
