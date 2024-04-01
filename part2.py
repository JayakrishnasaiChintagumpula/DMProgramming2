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
from sklearn.cluster import AgglomerativeClustering,KMeans
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

inertial_modelsse = {}
manual_modelsse = {}

def fit_kmeans(data, k):
    inertia_sse = []
    manual_sse = []
    for no_of_clusters in range(1, k + 1):
        k_means = KMeans(n_clusters=no_of_clusters)
        predictions = k_means.fit_predict(data)
        sse = {}
        for idx, pred in enumerate(predictions):
            diff_squared = (data[idx][0] - k_means.cluster_centers_[pred][0]) ** 2 + (data[idx][1] - k_means.cluster_centers_[pred][1]) ** 2
            try:
                sse[pred] += diff_squared
            except KeyError:
                sse[pred] = diff_squared

        inertia_sse.append(k_means.inertia_)
        manual_sse_value = 0
        for i in sse:
            manual_sse_value += sse[i]
        manual_sse.append(manual_sse_value)
        inertial_modelsse[no_of_clusters] = k_means.inertia_
        manual_modelsse[no_of_clusters] = manual_sse_value

    return inertia_sse, manual_sse


def compute():
    # ---------------------
    answers = {}

    """
    A. Call the make_blobs function with following parameters: (center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    x,label= datasets.make_blobs(n_samples=20, centers=5, center_box=(-20, 20), random_state=12)
    array_1 = x[:,0:2]
    array_2 = x[:,1:]
    dct = answers["2A: blob"] = [array_1, array_2, label]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
    # The function `fit_kmeans` is already modified above to return the SSE.
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C. Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    sse_val = fit_kmeans(array_1, 8)[1]
    sse_values = []
    for x,y in zip(range(1,9), sse_val):
        sse_values.append([x,y])
    plt.plot(np.array(sse_values)[:,1])
    plt.grid(True)
    plt.show()
    
    print(sse_values)
    dct = answers["2C: SSE plot"] = sse_values

    """
    D. Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    inertia_val = fit_kmeans(array_1, 8)[0]
    inertia_values = []
    for x,y in zip(range(1,9), inertia_val):
        inertia_values.append([x,y])
    
    dct = answers["2D: inertia plot"] = sse_values
    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
