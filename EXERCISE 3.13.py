#Exercise 3.13
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
