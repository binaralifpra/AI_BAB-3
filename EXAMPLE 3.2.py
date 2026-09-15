#Example 3.2 Python SVM Iris Classifications
from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
print(y)
clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[5.4, 3.2]])
print(p)
