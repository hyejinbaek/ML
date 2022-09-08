# kmeans clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = -2 * np.random.rand(100,2)
#print(X)
X1 = 1 + 2 * np.random.rand(50, 2)
#print(X1)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
plt.show()

from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

'''
#k means fit result
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10, n_jobs=None, precompute_distances='auto',random_state=None, tol=0.0001, verbose=0)
'''

# two centroid position
Kmean.cluster_centers_

plt.scatter(X[ : , 0], X[ : , 1], s = 50, c = 'b')
plt.scatter(-0.94665068, -0.97138368, s = 200, c = 'g', marker = 's')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()


