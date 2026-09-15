#Exercise 3.9
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = RandomForestRegressor()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("R2 Score: %f" % score)
