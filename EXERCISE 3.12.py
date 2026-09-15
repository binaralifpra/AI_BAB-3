#Exercise 3.12
from sklearn import linear_model
from sklearn.datasets import load_linnerud

linnerud = load_linnerud()
X = linnerud.data
y = linnerud.target

reg = linear_model.LinearRegression()
reg.fit(X, y)
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)
pred = reg.predict([X[0]])
print('Prediction: \n', pred)
