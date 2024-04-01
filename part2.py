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
    kmeans = KMeans(n_clusters=n_clusters, random_state=12)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    SSE = np.sum((data - centroids[labels]) ** 2)
    return SSE


def compute():
    # ---------------------
    answers = {}

    """
    A. Call the make_blobs function with following parameters: (center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    X, y_true = make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12)
    centers = np.array([X[y_true == i].mean(axis=0) for i in range(5)])
    dct = answers["2A: blob"] = [X, y_true, centers]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
    # The function `fit_kmeans` is already modified above to return the SSE.
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C. Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    sse_values = []
    for k in range(1, 9):
        sse = fit_kmeans(X, k)
        sse_values.append((k, sse))
    
    answers["2C: SSE plot"] = sse_values
    plt.plot([k[0] for k in sse_values], [k[1] for k in sse_values], 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method showing the optimal k')
    plt.grid(True)
    plt.show()

    dct = answers["2C: SSE plot"] = sse_values

    """
    D. Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    inertia_values = []
    for k in range(1, 9):
        kmeans = KMeans(n_clusters=k, random_state=12)
        kmeans.fit(X)
        inertia_values.append((k, kmeans.inertia_))
    
    answers["2D: inertia plot"] = inertia_values
    # Comparing SSE and inertia to see if the optimal k values agree.
    optimal_sse_k = min(sse_values, key=lambda t: t[1])[0]
    optimal_inertia_k = min(inertia_values, key=lambda t: t[1])[0]
    
    dct = answers["2C: SSE plot"] = inertia_values
    
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
