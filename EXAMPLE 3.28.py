# Example 3.28 LazyPredict (Simulasi LazyPredict menggunakan manual loop)
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

print("\n=== Part 2: Regressor (California Housing) ===")
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
