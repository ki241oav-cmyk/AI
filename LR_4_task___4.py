import numpy as np
from sklearn import covariance, cluster

names = np.array(['Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla', 'Meta', 'JPMorgan', 'Visa', 'Walmart', 'Procter & Gamble'])

np.random.seed(42)
X = np.random.randn(len(names), 100)

edge_model = covariance.GraphicalLassoCV()
edge_model.fit(X.T)

_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

print("\nКластери компаній на фондовому ринку:")
for i in range(num_labels + 1):
    cluster_names = names[labels == i]
    if len(cluster_names) > 0:
        print(f"Cluster {i+1} ==> {', '.join(cluster_names)}")