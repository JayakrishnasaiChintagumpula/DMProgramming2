from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    data,labels = dataset
    
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=0)
    kmeans.fit(standardized_data)

    predicted_labels = kmeans.labels_

    centroids =  kmeans.cluster_centers_
    distance = np.linalg.norm(standardized_data - centroids[predicted_labels], axis=1)
    sse = np.sum(distance**2)
    return sse

def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    np.random.seed(12)
    array_1, array_2, center  = make_blobs(n_samples=20, centers=5, center_box=(-20, 20), random_state=12)
    dct = answers["2A: blob"] = [array_1,array_2,center]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    sse_values = []
    for k in range(1, 9):
        sse = fit_kmeans((array_1,None), k)
        sse_values.append((k, sse))
    sse_values = [[k ,sse] for k,sse in sse_values]
    #Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, sse_values, '-o')
    plt.title('SSE as function of k)
    plt.xlabel('Number of clusters k')
    plt.ylabel('Value of metric')
    plt.grid(True)
    plt.show()

    dct = answers["2C: SSE plot"] = sse_values

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    # dct value has the same structure as in 2C
    inertia_values = []
    for k in range(1,9):
        kmeans = KMeans(n_clusters=k, random_state= 0, init= "random")
        kmeans.fit(X)
        
        inertia= kmeans.inertia_
        inertia_values.append((k,inertia))
    inertia_values = [[k, inertia] for k,inertia in inertia_values]

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia_values, '-o')
    plt.title('Inertia as function of k)
    plt.xlabel('Number of clusters k')
    plt.ylabel('Value of metric')
    plt.grid(True)
    plt.show()
        
    dct = answers["2D: inertia plot"] = inertia_values

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
