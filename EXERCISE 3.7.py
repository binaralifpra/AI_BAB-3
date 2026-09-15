#Exercise 3.7
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
f = plt.figure(1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel(cancer.feature_names[0])
plt.ylabel(cancer.feature_names[1])
plt.title('Original Breast Cancer Data')
f.show()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)
g = plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Data')
g.show()
