#Exercise 3.14
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
