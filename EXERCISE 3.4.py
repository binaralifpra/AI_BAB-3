#Exercise 3.4
import pandas as pd
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df[['mean radius', 'mean area', 'mean texture', 'mean smoothness']].hist()
plt.show()
