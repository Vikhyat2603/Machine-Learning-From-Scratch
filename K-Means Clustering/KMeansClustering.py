# This runs a matplotlib live animation, do not run inline

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.random.seed(1)

samples = 200 # Samples per cluster
c1 = np.random.randn(samples, 2) + [ 0,  0] # Make a cluster with centre 0,0
c2 = np.random.randn(samples, 2) + [ 4,  4] # Make a cluster with centre 4,4
c3 = np.random.randn(samples, 2) + [ 1,  6] # Make a cluster with centre 1,6
c4 = np.random.randn(samples, 2) + [-1, -9] # Make a cluster with centre -1,-9
points = np.concatenate((c1, c2, c3, c4)) # Combine clusters to get dataset

K = 4 # Number of clusters

centroids = np.random.randn(K, 2) + points.mean(axis=0) # Picks K random centroids near the dataset mean
classes = np.zeros(points.shape[0]) # Array initialised to store cluster/class for each point
clusters = np.zeros(K, dtype = np.object) # Array initialised to store points belonging to different clusters

plt.close("all") # Close all open plots
fig, ax = plt.subplots()

def animate(_, classes, clusters, centroids):
    ''' Classes, groups and centroids are passed as parameters to allow global access'''
    
    #Clear previous scatter plot
    ax.clear()
    
    #Classify every point to nearest centroid by making another array containing the cluster number for each point
    classes[:] = np.array([np.sum((p - centroids)**2, axis = 1).argmin() for p in points])
    
    #Dividing the points into their separate clusters
    clusters[:] = [points[np.where(classes == c)[0]] for c in range(K)]
      
    #Plot the points, coloured according to the clusters they belong to
    ax.scatter(points[:, 0], points[:, 1], c=classes)    
    
    #Plot the centroids shaped as diamonds, coloured according to clusters, with a red border
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='D', c=np.arange(K), edgecolors='red')
    
    #Calculate new centroid coordinates as : (mean of clusters' points' coordinates) if points are assigned to centroid else (old centroid coordinate)
    centroids[:] = [(cluster.mean(axis=0) if cluster.shape[0] else prevCentroid) for prevCentroid, cluster in zip(centroids, clusters)]
 
#Create an animation that calls the animate function with arguments every 1 second
ani = animation.FuncAnimation(fig, animate, fargs = (classes, clusters, centroids), interval=1000)

plt.show()