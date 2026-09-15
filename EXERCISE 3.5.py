#Exercise 3.5
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import joblib

X, y = load_iris(return_X_y=True)
clf = GaussianNB()
clf.fit(X, y)
joblib.dump(clf, 'naive_bayes_model.pkl')
clf2 = joblib.load('naive_bayes_model.pkl')
p = clf2.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)
