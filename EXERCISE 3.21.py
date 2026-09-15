# Exercise 3.21 (Simulasi LazyRegressor)
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
