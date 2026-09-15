import os

files = {
    "EXAMPLE 3.18.py": """#Example 3.18 Multiple Linear Regression
from sklearn import linear_model
import numpy as np
x = np.array([[0,3,5],[1,4,6],[2,5,7],[3,6,8],[4,7,9]])
y = np.array([3,5,5,6,7])
reg = linear_model.LinearRegression()
reg.fit(x, y)
print('Coefficients: \\n', reg.coef_)
print('Intercept: \\n', reg.intercept_)
pred = reg.predict([[5,8,10]])
print('Predition: \\n', pred)
""",
    "EXCERCISE 3.12.py": """#Exercise 3.12
from sklearn import linear_model
from sklearn.datasets import load_linnerud

linnerud = load_linnerud()
X = linnerud.data
y = linnerud.target

reg = linear_model.LinearRegression()
reg.fit(X, y)
print('Coefficients: \\n', reg.coef_)
print('Intercept: \\n', reg.intercept_)
pred = reg.predict([X[0]])
print('Prediction: \\n', pred)
""",
    "EXAMPLE 3.19.py": """#Example 3.19 Logistic Regression
import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[0],[1],[2],[3],[4],[5]])
y = np.array([1,2,3,30,32,31])
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.predict([[6]]))
print(clf.predict_proba([[6]]))
print(clf.score(X, y))
""",
    "EXAMPLE 3.20.py": """#Example 3.20 K-means Clustering
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2, 3], [1, 4, 2], [1, 0, 3], [10, 2, 4], [9, 4, 3], [11, 0, 2]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.predict([[12, 3, 1]]))
""",
    "EXCERCISE 3.13.py": """#Exercise 3.13
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
""",
    "EXAMPLE 3.21.py": """#Example 3.21 Semi-supervised Learning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading

X = np.array([[0,1],[1,1],[2,0],[3,1],[10,5],[11,6],[12,4],[13,5]])
y = np.array([0,0,0,0,1,1,1,1])
labels = np.full(8, -1.)
labels[0] = 0
labels[-1] = 1
print(labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
output_labels = label_spread.transduction_
print(output_labels)
""",
    "EXCERCISE 3.14.py": """#Exercise 3.14
import numpy as np
from sklearn.semi_supervised import LabelSpreading

X = np.array([[0,1],[1,1],[2,0],[3,1], [-1,0], [0,2], [10,5],[11,6],[12,4],[13,5], [14,4], [10,7]])
labels = np.full(12, -1.)
labels[0] = 0
labels[-1] = 1

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
output_labels = label_spread.transduction_
print(output_labels)
""",
    "EXCERCISE 3.15.py": """#Exercise 3.15
import numpy as np
from sklearn.semi_supervised import LabelSpreading

X = np.array([
    [0,1],[1,1],[2,0],[3,1], 
    [10,5],[11,6],[12,4],[13,5],
    [20,20], [21,21], [19,20], [20,19]
])
labels = np.full(12, -1.)
labels[0] = 0
labels[4] = 1
labels[8] = 2

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
output_labels = label_spread.transduction_
print(output_labels)
""",
    "EXAMPLE 3.22.py": """#Example 3.22 Q-Learning
import numpy as np
import pylab as plt
import networkx as nx

def showgraph(points_list):
    G=nx.Graph()
    G.add_edges_from(points_list)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G,pos)
    plt.show()

def createRmat(MATRIX_SIZE,points_list,goal):
    R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
    R *= -1
    for point in points_list:
        if point[1] == goal:
            R[point] = 100
        else:
            R[point] = 0
        if point[0] == goal:
            R[point[::-1]] = 100
        else:
            R[point[::-1]]= 0
    R[goal,goal]= 100
    return R

def available_actions(R, state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

def sample_next_action(available_act):
    next_action = int(np.random.choice(available_act,1))
    return next_action

def update(R, Q, current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    if (np.max(Q) > 0):
        return(np.sum(Q/np.max(Q)*100))
    else:
        return (0)

points_list = [(0,1),(1,2),(1,3),(2,4),(3,5),(3,6)]
goal = 6
# showgraph(points_list) # Un-comment to see plot
MATRIX_SIZE = 7
R = createRmat(MATRIX_SIZE,points_list,goal)
Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))
gamma = 0.8
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R,Q,current_state,action,gamma)
    scores.append(score)

print("Trained Q matrix:")
print(Q/np.max(Q)*100)

current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)
""",
    "EXCERCISE 3.16.py": """#Exercise 3.16
import numpy as np
import pylab as plt
import networkx as nx

def createRmat(MATRIX_SIZE,points_list,goal):
    R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
    R *= -1
    for point in points_list:
        if point[1] == goal:
            R[point] = 100
        else:
            R[point] = 0
        if point[0] == goal:
            R[point[::-1]] = 100
        else:
            R[point[::-1]]= 0
    R[goal,goal]= 100
    return R

def available_actions(R, state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

def sample_next_action(available_act):
    next_action = int(np.random.choice(available_act,1))
    return next_action

def update(R, Q, current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    if (np.max(Q) > 0):
        return(np.sum(Q/np.max(Q)*100))
    else:
        return (0)

# 8-state routing diagram
points_list = [(0,1),(1,2),(1,4),(2,3),(3,5),(4,6),(5,7),(6,7)]
goal = 7
MATRIX_SIZE = 8
R = createRmat(MATRIX_SIZE,points_list,goal)
Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))
gamma = 0.8
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R,Q,current_state,action,gamma)
    scores.append(score)

current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)
""",
    "EXAMPLE 3.23.py": """# Example 3.23 OpenAI Gym CartPole
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
""",
    "EXAMPLE 3.24.py": """# Example 3.24 Ensemble1.py
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = SVC()
eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svc', clf4)],
    voting='hard')

for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'SVM', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
""",
    "EXCERCISE 3.17.py": """# Exercise 3.17
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = SVC()
clf5 = KNeighborsClassifier()

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svc', clf4), ('knn', clf5)],
    voting='hard')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'SVM', 'KNN', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
""",
    "EXAMPLE 3.25.py": """# Example 3.25 Ensemble2.py
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor

X, y = load_diabetes(return_X_y=True)
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
reg4 = MLPRegressor()
reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)
reg4.fit(X, y)
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3), ('NN', reg4)])
print(ereg.fit(X, y))

xt = X[:20]
pred1 = reg1.predict(xt)
pred2 = reg2.predict(xt)
pred3 = reg3.predict(xt)
pred4 = reg4.predict(xt)
pred5 = ereg.predict(xt)

plt.figure()
plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
plt.plot(pred2, 'b^', label='RandomForestRegressor')
plt.plot(pred3, 'ys', label='LinearRegression')
plt.plot(pred4, 'mo', label='NeuralNetworks')
plt.plot(pred5, 'r*', ms=10, label='VotingRegressor')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Regressor predictions and their average')
plt.show()
""",
    "EXAMPLE 3.26.py": """# Example 3.26 Auto_ml_test.py
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
df_train, df_test = get_boston_dataset()
column_descriptions = {
    'MEDV': 'output',
    'CHAS': 'categorical'
}
ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
ml_predictor.train(df_train)
ml_predictor.score(df_test, df_test.MEDV)
""",
    "EXCERCISE 3.18.py": """# Exercise 3.18
from auto_ml import Predictor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['Target'] = california.target
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

column_descriptions = {
    'Target': 'output'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
ml_predictor.train(df_train)
ml_predictor.score(df_test, df_test.Target)
""",
    "EXAMPLE 3.27.py": """#Example 3.27 The PyCaret_demo.py
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris(as_frame=True)
iris.data['Target'] = iris.target
iris = iris.data
iris.head()
from pycaret import classification
classification.setup(data= iris, target='Target')
classification.compare_models()
""",
    "EXCERCISE 3.19.py": """# Exercise 3.19
import pandas as pd
from sklearn import datasets
from pycaret import classification

cancer = datasets.load_breast_cancer(as_frame=True)
df = cancer.data
df['Target'] = cancer.target

classification.setup(data=df, target='Target')
classification.compare_models()
""",
    "EXAMPLE 3.28.py": """# Example 3.28 LazyPredict
# !pip install lazypredict

import lazypredict
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Part 1, 2, 3: Classifier
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.25,random_state =1)
clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

plt.figure(figsize=(10, 5))
plt.plot(models.index, models['Accuracy'])
plt.title('Classifier Accuracy')
plt.show()

# Part 4, 5: Regressor
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.1,random_state =1)
reg = LazyRegressor()
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

plt.figure(figsize=(10, 5))
plt.plot(models.index, models['R-Squared'],'-s')
plt.title('Regressor R-Squared')
plt.show()
""",
    "EXCERCISE 3.20.py": """# Exercise 3.20
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
""",
    "EXCERCISE 3.21.py": """# Exercise 3.21
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
reg = LazyRegressor()
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
"""
}

for name, content in files.items():
    with open(name, 'w', encoding='utf-8') as f:
        f.write(content)

print("Files generated successfully!")
