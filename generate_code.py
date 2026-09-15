import os

files = {
    "EXAMPLE 3.1.py": """#Example 3.1 Python SVM Classifications
from sklearn import svm
X = [[170, 70, 10], [180, 80,12], [170, 65, 8],[160, 55, 7]]
y = [0, 0, 1, 1] 
clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[160, 60, 7]])
print(p)
""",
    "EXCERCISE 3.1.py": """#Exercise 3.1
from sklearn import svm
X = [[170, 70, 10], [180, 80, 12], [170, 65, 8], [160, 55, 7], [175, 75, 11], [155, 50, 6]]
y = [0, 0, 1, 1, 0, 1] 
clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[160, 60, 7]])
print(p)
""",
    "EXAMPLE 3.2.py": """#Example 3.2 Python SVM Iris Classifications
from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
print(y)
clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[5.4, 3.2]])
print(p)
""",
    "EXCERCISE 3.2.py": """#Exercise 3.2
from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target
print(y)
clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[5.4, 3.2]])
print(p)
""",
    "EXAMPLE 3.3.py": """#Example 3.3 Python SVM Iris CSV Classifications
from sklearn import svm, datasets
import pandas as pd
df = pd.read_csv('iris.csv')
X = df.values[:,:2]
s = df['species']
d = dict([(y,x) for x,y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]
clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[5.4, 3.2]])
print(p)
""",
    "EXAMPLE 3.4.py": """#Example 3.4 Python SVM Iris URL Classifications
from sklearn import svm, datasets
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
print(df.shape)
print(df.head(10))
print(df.tail(10))
print(df.describe())
print(df.isna().sum().sum())
df = df.dropna()
print(df.groupby('species').size())
df.hist()
pyplot.show()
scatter_matrix(df)
pyplot.show()
X = df.values[:,:2]
s = df['species']
d = dict([(y,x) for x,y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]
clf = svm.SVC()
clf.fit(X, y)
p = clf.predict([[5.4, 3.2]])
print(p)
""",
    "EXCERCISE 3.3.py": """#Exercise 3.3
from matplotlib import pyplot as plt
import pandas as pd
df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter plot of first two features')
plt.show()
""",
    "EXAMPLE 3.5.py": """#Example 3.5 Breast Cancer Data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer['feature_names'])
print(cancer['data'])
print(cancer.target_names)
""",
    "EXAMPLE 3.6.py": """#Example 3.6 Python SVM Breast Cancer Classifications
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
cancer = load_breast_cancer()
X = cancer.data 
y = cancer.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
clf = SVC()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['predicted_cancer','predicted_healthy'])
print(confusion)
print(classification_report(y_test, y_predict))
""",
    "EXCERCISE 3.4.py": """#Exercise 3.4
import pandas as pd
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df[['mean radius', 'mean area', 'mean texture', 'mean smoothness']].hist()
plt.show()
""",
    "EXAMPLE 3.7.py": """#Example 3.7 Naive Bayes Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
print(X)
clf = GaussianNB()
clf.fit(X, y)
p = clf.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)
""",
    "EXCERCISE 3.5.py": """#Exercise 3.5
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
""",
    "EXAMPLE 3.8.py": """#Example 3.8 LinearDiscriminantAnalysis Classification
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
print(X)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))
""",
    "EXCERCISE 3.6.py": """#Exercise 3.6
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X, y = make_classification(n_samples=2000, n_features=6, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
print(X)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0, 0, 0]]))
""",
    "EXAMPLE 3.9.py": """#Example 3.9 Principal Component Analysis
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
print(X)
clf = PCA()
clf.fit(X, y)
print(clf.explained_variance_ratio_)
print(clf.singular_values_)
""",
    "EXAMPLE 3.10.py": """#Example 3.10 Principal Component Analysis Iris
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
f = plt.figure(1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel('sepals length')
plt.ylabel('sepals width')
plt.title('Original Data')
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
""",
    "EXCERCISE 3.7.py": """#Exercise 3.7
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
""",
    "EXAMPLE 3.11.py": """#Example 3.11 Decision Tree Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
N = y_test.shape[0]
C = (y_test == y_pred).sum()
print("Total points: %d Correctly labeled points : %d" %(N,C))
""",
    "EXCERCISE 3.8.py": """#Exercise 3.8
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
N = y_test.shape[0]
C = (y_test == y_pred).sum()
print("Total points: %d Correctly labeled points : %d" %(N,C))
""",
    "EXAMPLE 3.12.py": """#Example 3.12 Random Forest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Total points: %d Correctly labeled points : %d" %(y_test.shape[0],(y_test == y_pred).sum()))
""",
    "EXCERCISE 3.9.py": """#Exercise 3.9
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = RandomForestRegressor()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("R2 Score: %f" % score)
""",
    "EXAMPLE 3.13.py": """#Example 3.13 K-Nearest Neighbors Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Total points: %d Correctly labeled points : %d" %(y_test.shape[0],(y_test == y_pred).sum()))
""",
    "EXAMPLE 3.14.py": """#Example 3.14 Comparison of Different Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = [ "SVM", "Naive Bayes", "LDA", "QDA", "Decision Tree", "Random Forest", "Nearest Neighbors", "Neural Networks"]
classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000)
]

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name +": " + str(score))
""",
    "EXCERCISE 3.10.py": """#Exercise 3.10
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

names = ["SVR", "Decision Tree", "Random Forest", "Nearest Neighbors", "Neural Networks"]
classifiers = [
    SVR(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    KNeighborsRegressor(),
    MLPRegressor(max_iter=1000)
]
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name + ": " + str(score))
""",
    "EXAMPLE 3.15.py": """#Example 3.15 Linear Regression
import matplotlib.pyplot as plt
from scipy import stats
x = [0,1,2,3,4]
y = [3,5,5,6,7]
slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope: ", slope)
print("intercept: ", intercept)
def myfunc(x):
    return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
""",
    "EXCERCISE 3.11.py": """#Exercise 3.11
import matplotlib.pyplot as plt
from scipy import stats
x = [0,1,2,3,4,5,6,7,8,9,10]
y = [3,5,5,6,7,8,9,9,10,12,11]
slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope: ", slope)
print("intercept: ", intercept)
def myfunc(x):
    return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Linear Regression Example')
plt.legend(['Data points', 'Best fit line'])
plt.grid(True)
plt.show()
""",
    "EXAMPLE 3.15a.py": """#Example 3.15a Linear Regression
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
x = [0,1,2,3,4]
y = [3,5,5,6,7]
x1=sm.add_constant(x)
model = sm.OLS(y,x1)
results = model.fit()
print (results.params)
print (results.summary())
y_pred=results.predict(x1)
plt.scatter(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x,y_pred, "r")
plt.show()
""",
    "EXAMPLE 3.16.py": """#Example 3.16 Polynomial Regression
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
x = [0,1,2,3,4,5]
y = [3,8,6,6,7,3]
mymodel = np.poly1d(np.polyfit(x, y, 3))
print(mymodel)
myline = np.linspace(0, 5, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
""",
    "EXAMPLE 3.17.py": """#Example 3.17 Least Squares Fitting
import numpy as np
import scipy.optimize as optimization
x = np.array([0,1,2,3,4,5])
y = np.array([100,90,60,30,10,1])
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
popt, pcov = optimization.curve_fit(func, x, y)
print ("Best fit a b c: ",popt)
print ("Best fit covariance: ",pcov)
"""
}

for name, content in files.items():
    with open(name, 'w', encoding='utf-8') as f:
        f.write(content)

print("Files generated successfully!")
