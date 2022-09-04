'''
Created by: Vicky Parmar
2020-02-06 11:35:33
'''

# %%
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# %%
# Generating sample data
# cen = [[2,8], [0,8], [1,8]]
X, lab = make_blobs(n_samples=750, centers=4, cluster_std=0.6, random_state=0)

X = StandardScaler().fit_transform(X)   # Standardizing data

# %%
# Finding EPS
neigh = NearestNeighbors(2)
nbrs = neigh.fit(X)
dist, ind = nbrs.kneighbors(X)

dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist)

'Still need to play around with min_sanmples'

# %%
# Implementing DBSCAN
DBSCAN()
db = DBSCAN(eps=0.13, min_samples=7, metric='euclidean')
db.fit(X)
clusters = db.labels_

colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
