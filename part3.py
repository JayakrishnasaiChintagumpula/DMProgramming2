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
from scipy.cluster.hierarchy import dendrogram, linkage #
from scipy.spatial.distance import pdist

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
    data_points = mat_data['X']

    # Perform hierarchical clustering using the "single" linkage method
    Z = linkage(data_points, 'single')
    
    plt.figure(figsize=(25, 10))
    dendo=dendrogram(Z)
    plt.savefig("3B_dendrogram.pdf")
    plt.show()

    # Answer: the return value of the dendogram function, dicitonary
    answers["3B: linkage"] = Z
    answers["3B: dendogram"] = dendo

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """

    # Answer type: integer
    cluster_I = [8, 2, 13]
    cluster_J = [1, 9]

    # Combine indices from clusters I and J
    combined_indices = cluster_I + cluster_J

    # Extract the combined data points
    combined_data_points = data_points[combined_indices, :]

    # Calculate the Euclidean distance matrix for the combined data points
    distance_matrix_combined = pdist(combined_data_points, metric='euclidean')

    # Perform hierarchical clustering with the single linkage method on the combined distance matrix
    Z_combined = linkage(distance_matrix_combined, method='single')

    # Extract dissimilarities from the linkage matrix for combined clusters I and J
    dissimilarities_combined = Z_combined[:, 2]

    # The final merge dissimilarity among the combined clusters I and J
    final_merge_dissimilarity = dissimilarities_combined[-1]

    # Find the iteration in the full dataset's linkage matrix Z where this dissimilarity occurs
    iteration_found = None
    for i, row in enumerate(Z):
        if np.isclose(row[2], final_merge_dissimilarity, atol=1e-04):
            iteration_found = i
            break

    # Output the iteration where clusters I and J would have their final merge in the full dataset context
    print(iteration_found)
    answers["3C: iteration"] = iteration_found

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
    clusters = [[i] for i in range(len(data_points))]

    # Iterate through the linkage matrix up to the specified iteration
    for i in range(iteration_found + 1):
        # Each row in the linkage matrix represents a merge operation.
        merge_info = Z[i]
        idx1, idx2 = int(merge_info[0]), int(merge_info[1])  # Indices of the clusters to be merged.
        
        # The new cluster is the union of the two clusters.
        new_cluster = clusters[idx1] + clusters[idx2]
        
        # Add the new cluster to the list of clusters.
        clusters.append(new_cluster)
        
        # Remove the old clusters.
        clusters[idx1] = []
        clusters[idx2] = []

    # Remove empty clusters and sort each cluster's indices for readability.
    clusters = [sorted(cluster) for cluster in clusters if cluster]

    print(clusters)
    answers["3E: clusters"] = clusters

    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = "True, The analysis indicates that the rich get richer phenomenon is observed in the hierarchical clustering process for this dataset. This conclusion is based on the observation that some clusters grow significantly larger than the average cluster size, especially towards the later stages of the clustering process. The sizes of the last few clusters formed demonstrate a continuous increase, which is characteristic of the rich gets richer phenomenon."

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
