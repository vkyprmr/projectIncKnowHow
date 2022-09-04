"""
Created by: Vicky Parmar
2020-02-05 11:23:45
"""

# Imports
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate our own data
cluster_centers = [[1,2,3], [5,6,7], [4,8,9]]
X, _ = make_blobs(n_samples=250, centers=cluster_centers, cluster_std=0.75)


# Fitting the data
ms = MeanShift()
ms.fit(X)
pred_centers = ms.cluster_centers_


# Plotting a 3d- graph
# %matplotlib qt    # Interactive window
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0],X[:,1],X[:,2], marker='o')

ax.scatter(pred_centers[:,0], pred_centers[:,1], pred_centers[:,2], marker='x',
            color='red', s=300, linewidth=5, zorder=10)



