#Exercise 3.15
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
