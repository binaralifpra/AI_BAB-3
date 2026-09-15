import os

files = {
    "EXAMPLE 3.26.py": """# Example 3.26 Auto_ml_test.py (Diganti menggunakan FLAML karena auto_ml sudah usang)
from flaml import AutoML
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Menggunakan California Housing karena dataset Boston sudah dihapus dari Scikit-Learn
california = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.2, random_state=42)

automl = AutoML()
# Dibatasi 5 detik agar cepat selesai untuk contoh
automl.fit(X_train, y_train, task="regression", time_budget=5, verbose=0)
print("Best ML model:", automl.best_estimator)
print("Score (R2):", automl.score(X_test, y_test))
""",

    "EXCERCISE 3.18.py": """# Exercise 3.18 (Diganti menggunakan FLAML karena auto_ml sudah usang)
from flaml import AutoML
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

california = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.2, random_state=42)

automl = AutoML()
automl.fit(X_train, y_train, task="regression", time_budget=5, verbose=0)
print("Best ML model:", automl.best_estimator)
print("Score (R2):", automl.score(X_test, y_test))
""",

    "EXAMPLE 3.27.py": """# Example 3.27 The PyCaret_demo.py (Simulasi Compare Models tanpa PyCaret)
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X, y = iris.data, iris.target

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

results = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results.append({'Model': name, 'Accuracy': cv_scores.mean(), 'Std Dev': cv_scores.std()})

df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print("=== Simulasi PyCaret compare_models() ===")
print(df)
""",

    "EXCERCISE 3.19.py": """# Exercise 3.19 (Simulasi Compare Models tanpa PyCaret)
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

results = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results.append({'Model': name, 'Accuracy': cv_scores.mean(), 'Std Dev': cv_scores.std()})

df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print("=== Simulasi PyCaret compare_models() Breast Cancer ===")
print(df)
""",

    "EXAMPLE 3.28.py": """# Example 3.28 LazyPredict (Simulasi LazyPredict menggunakan manual loop)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

print("=== Part 1: Classifier (Iris) ===")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1)

classifiers = {
    'RandomForestClassifier': RandomForestClassifier(random_state=1),
    'LogisticRegression': LogisticRegression(max_iter=200, random_state=1),
    'SVC': SVC(random_state=1),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=1)
}

clf_results = []
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    clf_results.append({'Model': name, 'Accuracy': acc})

df_clf = pd.DataFrame(clf_results).set_index('Model').sort_values('Accuracy', ascending=False)
print(df_clf)

plt.figure(figsize=(10, 5))
plt.plot(df_clf.index, df_clf['Accuracy'], marker='o')
plt.title('Classifier Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\\n=== Part 2: Regressor (California Housing) ===")
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=1)

regressors = {
    'RandomForestRegressor': RandomForestRegressor(random_state=1, n_estimators=50),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1)
}

reg_results = []
for name, model in regressors.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    reg_results.append({'Model': name, 'R-Squared': r2})

df_reg = pd.DataFrame(reg_results).set_index('Model').sort_values('R-Squared', ascending=False)
print(df_reg)

plt.figure(figsize=(10, 5))
plt.plot(df_reg.index, df_reg['R-Squared'], '-s', color='orange')
plt.title('Regressor R-Squared')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
""",

    "EXCERCISE 3.20.py": """# Exercise 3.20 (Simulasi LazyClassifier)
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifiers = {
    'RandomForestClassifier': RandomForestClassifier(random_state=1),
    'LogisticRegression': LogisticRegression(max_iter=5000, random_state=1),
    'SVC': SVC(random_state=1),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=1)
}

results = []
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results.append({'Model': name, 'Accuracy': acc})

df = pd.DataFrame(results).set_index('Model').sort_values('Accuracy', ascending=False)
print("=== Simulasi LazyClassifier Wine ===")
print(df)
""",

    "EXCERCISE 3.21.py": """# Exercise 3.21 (Simulasi LazyRegressor)
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressors = {
    'RandomForestRegressor': RandomForestRegressor(random_state=1),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1)
}

results = []
for name, model in regressors.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    results.append({'Model': name, 'R-Squared': r2})

df = pd.DataFrame(results).set_index('Model').sort_values('R-Squared', ascending=False)
print("=== Simulasi LazyRegressor Diabetes ===")
print(df)
"""
}

for name, content in files.items():
    with open(name, 'w', encoding='utf-8') as f:
        f.write(content)

print("Files rewritten successfully!")
