'''
Created by: Vicky Parmar
2020-02-02 18:32:14
'''


# In[]:
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# In[]:
# Reading from excel and visualisation
s1 = pd.read_excel('s1.xlsx')   # Read any file via read_excel, read_csv, etc

plt.scatter(s1.X, s1.Y)     # Visualize using matplotlib

# In[]:
# Standardizing data    # Standardizing always proves beneficial and you can
#                         also invert transformation back to original values
ss = StandardScaler()
s1_std = pd.DataFrame(ss.fit_transform(s1), columns=['X','Y'])

# In[]:
# Looking for optimal 'k' using Elbow ('L')-curve
distortions = []
K = range(1,31)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(s1_std)
    kmeanModel.fit(s1_std)
    distortions.append(sum(np.min(cdist(s1_std, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / s1_std.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k s1')
plt.show()

# In[]:
# K=15 'Using optimal k found in the above step to find clusters'
kmeans = KMeans(n_clusters=15)
kmeans.fit(s1_std)
centroid = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting results
s1_std.plot(kind = 'scatter', x='X', y='Y', c=kmeans.labels_, cmap='rainbow', s=40)
plt.scatter(centroid[:,0], centroid[:,1] ,color='black', marker='*', s=30)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter_Clusters_15")
plt.grid()