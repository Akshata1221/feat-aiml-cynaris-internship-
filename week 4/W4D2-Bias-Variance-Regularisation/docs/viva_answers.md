# Viva answers

1. I compared L1 and L2 regularised logistic regression with GridSearchCV and RandomizedSearchCV, using stratified cross-validation and MLflow. Scaling is inside a pipeline to avoid leakage.
2. The hardest part was balancing training fit with generalisation. I used cross-validation and a learning curve to identify the gap between training and validation scores.
3. With another day, I would evaluate additional models and use repeated cross-validation or Optuna for more efficient tuning.
