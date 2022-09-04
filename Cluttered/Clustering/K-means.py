"""
Created by: Vicky Parmar
2020-02-02 18:32:14
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


# Reading from excel and visualisation
df = pd.read_excel('df.xlsx')   # Read any file via read_excel, read_csv, etc

plt.scatter(df.X, df.Y)     # Visualize using matplotlib

'''
To read from a text file with space as delimiter

def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x,y

x,y = Read_Two_Column_File('A:/Clustering/birch1.txt')

'''


# Standardizing data    # Standardizing always proves beneficial and you can
#                         also invert transformation back to original values
ss = StandardScaler()
df_std = pd.DataFrame(ss.fit_transform(df), columns=['X','Y'])


# Looking for optimal 'k' using Elbow ('L')-curve
distortions = []
K = range(1,31)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_std)
    distortions.append(sum(np.min(cdist(df_std, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df_std.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k df')
plt.show()


# K=15 'Using optimal k found in the above step to find clusters'
kmeans = KMeans(n_clusters=15)
kmeans.fit(df_std)
centroid = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting results
df_std.plot(kind = 'scatter', x='X', y='Y', c=kmeans.labels_, cmap='rainbow', s=40)
plt.scatter(centroid[:,0], centroid[:,1] ,color='black', marker='*', s=30)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter_Clusters_15")
plt.grid()
