#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:13:03 2020

@author: apple
"""
#BB Supermarket Customer Segmentation using K-means Clustering
#Using The sklearn(scikit-learn) package

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# X contains training data samples
X=[[3,280],[5,320],[8,80],[2,225],[5,80],[6,265],[1,71],[5,235],[7,98],[4,300],[2,210],[3,70],[5,99],[3,210],[3,125],[7,280]]

# Number of clusters k
k=4
kmeans = KMeans(n_clusters=k)

# Fitting the input data
kmeans = kmeans.fit(X)

# Centroid values
Xarr=np.array(X)
centroids = kmeans.cluster_centers_
print("\nCluster Centers are :\n",centroids)

# Predicting the cluster labels
labels = kmeans.predict(X)
print("\nCluster Labels are:\n",labels)

colors = ['r', 'g', 'b', 'y', 'c', 'm']

fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(centroids[i, 0], centroids[i, 1], marker='*', s=200, c=colors[i])
           
# predicting class of input pattern [4,200]
xnew=[[4,200]]
print("\nPattern ", xnew, "belongs to cluster #",kmeans.predict(xnew))