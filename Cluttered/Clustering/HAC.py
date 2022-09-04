'''
Created by: Vicky Parmar
2020-02-07 10:13:22
'''

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

# %%
# Reading data
filepath = 'A:/Clustering/Mall_Customers.csv'
df = pd.read_csv(filepath)
X = df.iloc[:,[3,4]].values

# %%
# Finding optimal number of clusters
''' You can plot a dendogram. Look for the maximum vertical distance witout any clusters.
    Plot two horizontal lines at both extreme ends. The number of vertical lines between them is the optimal number of clusters. '''

dendogram = shc.dendrogram(shc.linkage(X, method='ward'))

# %%
# HAC model
hac = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
hac.fit(X)
labels = hac.labels_

# %%
# Plotting the results
plt.scatter(X[labels==0, 0], X[labels==0, 1], marker='o')
plt.scatter(X[labels==1, 0], X[labels==1, 1], marker='o')
plt.scatter(X[labels==2, 0], X[labels==2, 1], marker='o')
plt.scatter(X[labels==3, 0], X[labels==3, 1], marker='o')
plt.scatter(X[labels==4, 0], X[labels==4, 1], marker='o')

# %%
