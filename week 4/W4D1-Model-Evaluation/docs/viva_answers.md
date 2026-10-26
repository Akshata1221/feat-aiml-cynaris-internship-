# Viva preparation

**1. What did you build and why?**  I built a model evaluation pipeline comparing Logistic Regression and Decision Tree using precision, recall, F1, ROC AUC, stratified cross-validation and learning curves. I used stratified folds for balanced evaluation, a scaling pipeline to prevent leakage, and MLflow to track metrics.

**2. Hardest part and solution?**  Avoiding data leakage while cross-validating was challenging. I put scaling inside the model pipeline and used the held-out test set only after training and cross-validation.

**3. What would you improve with one more day?**  I would compare more algorithms, tune classification thresholds, examine class-specific error costs and add confidence intervals.

**Key concepts:** Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2PR/(P+R). ROC AUC measures ranking ability across thresholds. Overfitting: training score much higher than validation; underfitting: both low.
