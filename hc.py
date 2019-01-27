# Hierarchical Clustering

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values


# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Clusters')
plt.ylabel('Euclidean Distance')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], c='red', label='Cluster1')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], c='blue', label='Cluster2')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], c='green', label='Cluster3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], c='yellow', label='Cluster4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], c='pink', label='Cluster5')
#plt.scatter(hc.cluster_centers_[:,0], hc.cluster_centers_[:,1], s=200, c='black', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()