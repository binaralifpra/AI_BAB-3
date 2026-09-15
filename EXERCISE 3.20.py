# Exercise 3.20 (Simulasi LazyClassifier)
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
