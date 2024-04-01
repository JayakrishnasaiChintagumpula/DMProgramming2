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

"""
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def data_index_function(I, J, distance_matrix, Z, dissimilarities):
    # Check if we have already calculated the dissimilarity
    key = frozenset(I), frozenset(J)
    if key in dissimilarities:
        return dissimilarities[key]
    
    # Calculate dissimilarity if not already done
    # Extract the distances between points in cluster I and cluster J
    cluster_I_distances = distance_matrix[np.ix_(I, J)]
    
    # The single linkage dissimilarity is the minimum of these distances
    single_link_dissimilarity = np.min(cluster_I_distances)
    
    # Cache this dissimilarity
    dissimilarities[key] = single_link_dissimilarity
    
    return single_link_dissimilarity


def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierachal_toy_data.mat” using the scipy.io.loadmat function.
    """

    # return value of scipy.io.loadmat()
    mat_data = io.loadmat('hierarchical_toy_data.mat')
    answers["3A: toy data"] = mat_data

    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    """

    # Answer: NDArray
    Z = linkage(data_points, 'single')

    # Generate the dendrogram and capture its return value, which contains the structure of the dendrogram.
    plt.figure(figsize=(10, 7))
    dendrogram_return_value = dendrogram(Z)
    plt.close()  # Close the plot to not display it again here, as we're focusing on the structure.

    # Now, dendrogram_return_value contains the necessary information. Let's explicitly check for the keys.
    required_keys = {'icoord', 'dcoord', 'color_list', 'leaves', 'ivl'}
    present_keys = set(dendrogram_return_value.keys())

    # Verify that all required keys are present
    missing_keys = required_keys - present_keys

    # Output the present keys and any missing keys
    present_keys
    answers["3B: linkage"] = Z

    # Answer: the return value of the dendogram function, dicitonary
    answers["3B: dendogram"] = present_keys

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """

    # Answer type: integer
    answers["3C: iteration"] = -1

    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """
    # Answer type: a function defined above
    answers["3D: function"] = data_index_function

    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [{0,1,2},{3,4},{5},{6},…],  that were available when the two clusters in part 2.D were merged.
    """

    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    answers["3E: clusters"] = [{0, 0}, {0, 0}]

    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = ""

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
