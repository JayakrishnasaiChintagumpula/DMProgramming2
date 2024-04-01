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
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data, n_clusters, linkage='ward'):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit(data_scaled)
    return model.labels_

def fit_modified(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    Z = linkage(data_scaled, 'ward')
    diff = np.diff(Z[:, 2])
    elbow = np.argmax(diff)
    cutoff_distance = Z[elbow, 2]
    model = AgglomerativeClustering(distance_threshold=cutoff_distance, n_clusters=None)
    model.fit(data_scaled)
    return model.labels_, cutoff_distance


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    # Load the datasets
    noisy_circles = datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=100, noise=.05, random_state=random_state)
    blobs_varied = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    aniso_data, aniso_labels = datasets.make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = (np.dot(aniso_data, transformation), aniso_labels)

    blobs = datasets.make_blobs(n_samples=100, random_state=random_state)

    # Perform hierarchical clustering on the datasets with different linkage criteria
    datasets_list = [noisy_circles, noisy_moons, blobs_varied, aniso, blobs]
    linkage_types = ['ward', 'complete', 'average', 'single']
    datasets_labels = {}

    # Hierarchical clustering
    for i, dataset in enumerate(datasets_list):
        dataset_labels = {}
        for linkage in linkage_types:
            labels = fit_hierarchical_cluster(dataset[0], 2, linkage)
            dataset_labels[linkage] = labels
        datasets_labels[i] = dataset_labels
    dct = answers["4A: datasets"] = {'nc': noisy_circles,'nm': noisy_moons,'bvv': blobs_varied,'add': aniso,'b': blobs}

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations (see 1.C)
    datasets_dict = {
    'noisy_circles': datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state),
    'noisy_moons': datasets.make_moons(n_samples=100, noise=.05, random_state=random_state),
    'blobs': datasets.make_blobs(n_samples=100, random_state=random_state),
    # Applying a transformation to create an anisotropic dataset
    'aniso': lambda: ((np.dot(datasets.make_blobs(n_samples=100, centers=3, random_state=42)[0], [[0.6, -0.6], [-0.4, 0.8]]), datasets.make_blobs(n_samples=100, centers=3, random_state=random_state)[1])),
    'varied': lambda: datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state),
    }
    linkage_methods = ['ward', 'complete', 'average', 'single']

    fig, axs = plt.subplots(len(linkage_methods), len(datasets_dict), figsize=(20, 15))

    for i, linkage in enumerate(linkage_methods):
        for j, (dataset_name, dataset) in enumerate(datasets_dict.items()):
            # Some datasets are defined by functions to apply transformations
            if callable(dataset):
                data, labels = dataset()
            else:
                data, labels = dataset

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            cluster = AgglomerativeClustering(n_clusters=2, linkage=linkage)
            cluster.fit(data_scaled)

            ax = axs[i][j]
            ax.scatter(data_scaled[:, 0], data_scaled[:, 1], c=cluster.labels_, cmap='viridis', s=50, alpha=0.6)
            ax.set_title(f'{dataset_name} - {linkage}')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    dct = answers["4B: cluster successes"] = ["Not all Clusters are correctly clustered in the hierarchical clustering, but this is better than k_means"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    # dct is the function described above in 4.C
    fig, axs = plt.subplots(len(datasets_dict), 1, figsize=(10, 20))

    # Loop over datasets and apply the modified function
    for i, (dataset_name, dataset) in enumerate(datasets_dict.items()):
        # Unpack the dataset
        data, true_labels = dataset() if callable(dataset) else dataset

        # Apply the modified hierarchical clustering function
        labels, cutoff_distance = fit_modified(data)

        # Plotting the results
        ax = axs[i] if len(datasets_dict) > 1 else axs
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
        ax.set_title(f'{dataset_name} - Cutoff: {cutoff_distance:.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()

    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
