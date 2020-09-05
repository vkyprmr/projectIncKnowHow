#Coder: Vicky

#Imports
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

#data
data = xyz

#determining optimal "k"
distortions = []
k = range(1, 15)
for k in k:
    model = KMeans(n_clusters=k)
    model.fit(data)
    distortions.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1))/data.shape[0])
    
#Plot the elbow curve
plt.plot(k, distortions)
plt.xlabel('k')
plt.ylabel('distance')
plt.title('elbow curve')
plt.show()


#Time-series clustering by extracting features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh import extract_features, extract_relevant_features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Feature extraction
settings = ComprehensiveFCParameters()
fc_parameters = {
    "list of features":None #see documentation for features and also how to define your own feature
    }

data_features = extract_features(data, default_fc_parameters=fc_parameters, column_id='Nr', column_sort='Period') #see documentation for syntax explaination
#see documentation for extract_relevant_features

#Standardize
ss = StandardScaler()
data_features = pd.DataFrame(data_features)
data_std = ss.fit_transform(data_features, columns=data_features.columns)

#PCA
p = PCA(n_components=2)
pC = pca.fit_transform(data_std)
pC = pd.DataFrame(pC, columns=['pc1', 'pc2'])
#pC = pd.concat([pC, y], axis=1)

#Visualization of labeled data acc to categories
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_title('pC')
abc = ['list']
colors = ['list of colors']
for abc, color in zip(abc, colors):
    indicesToKeep = pC.abc == abc
    ax.scatter(pC.loc[indicesToKeep, 'pc1'],
               pC.loc[indicesToKeep, 'pc2'],
               c = color, s = 50) #see documentation for markers
ax.legend(abc)
ax.grid()


#Clustering
'''
KMeans
'''
pC = pC.drop(['y'], axis=1)
c = KMeans(n_clusters=6)
c.fit(pC)
centroids = c.cluster_cemters_
labels = c.labels_

pC.plot(kind = 'scatter', x='pc1', y='pc2', c=c.labels_, cmap='rainbow')
plt.scatter(centroid[:,0], centroid[:,1], color='k', marker='*')
plt.grid()
plt.show()


















