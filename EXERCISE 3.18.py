# Exercise 3.18 (Diganti menggunakan FLAML karena auto_ml sudah usang)
from flaml import AutoML
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

california = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.2, random_state=42)

automl = AutoML()
automl.fit(X_train, y_train, task="regression", time_budget=5, verbose=0)
print("Best ML model:", automl.best_estimator)
print("Score (R2):", automl.score(X_test, y_test))
