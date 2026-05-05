import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data 

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.title('Кластеризація наборів даних Iris (K-Means)')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()