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
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans():
    return None


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    dct = answers["1A: datasets"] = datasets
    random_state=42;
    datasets={}

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    nc = datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state)
    datasets['nc'] = nc

    nm = datasets.make_moons(n_samples=100, noise=.05, random_state=random_state)
    datasets['nm'] = nm

    bvv = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    datasets['bvv'] = bvv

    add = datasets.make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]  # Transformation matrix
    add[0] = np.dot(add[0], transformation)
    datasets['add'] = add

    b = datasets.make_blobs(n_samples=100, random_state=random_state)
    datasets['b'] = b

    
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    answers["1A: datasets"] = datasets
    answers = compute()
    return answers

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans
    def fit_kmeans(data, n_clusters):
        scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    kmeans = cluster.KMeans(n_clusters = n_clusters, init='random')
    kmeans.fit(standardized_data)

    predicted_labels = kmeans.labels_

    return predicted_labels
    answers['1B: fit_kmeans'] = fit_kmeans

    
    
        


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {"xy": [3,4], "zx": [2]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["xy"]
    def create_cluster_plots(datasets, fit_kmeans):

        k_values = [2, 3, 5, 10]

        cluster_successes = {}
        cluster_failures = []

        for dataset_abbr, (data, _) in datasets.items():
            fig, axs = plt.subplots(len(k_values), len(datasets), figsize=(15, 12))

            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(data)

            for i, k in enumerate(k_values):
                predicted_labels = fit_kmeans(standardized_data, k)

                ax = axs[i, datasets.keys().index(dataset_abbr)]
                ax.scatter(standardized_data[:, 0], standardized_data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'Dataset: {dataset_abbr}, k={k}')
                ax.set_xticks([])
                ax.set_yticks([])

                if len(np.unique(predicted_labels)) == k:
                    if dataset_abbr not in cluster_successes:
                        cluster_successes[dataset_abbr] = []
                    cluster_successes[dataset_abbr].append(k)
                else:
                    cluster_failures.append(dataset_abbr)

            plt.tight_layout()

            plt.savefig(f'cluster_plots_{dataset_abbr}.pdf')
            plt.close()

        return cluster_successes, cluster_failures

    cluster_successes, cluster_failures = create_cluster_plots(answers["1A: datasets"], answers["1B: fit_kmeans"])

    answers["1C: cluster successes"] = cluster_successes
    answers["1C: cluster failures"] = cluster_failures

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = [""]
    def analyze_initialization_sensitivity(datasets, fit_kmeans, num_iterations=5):

        k_values = [2, 3]

        sensitive_datasets = []

        for dataset_abbr, (data, _) in datasets.items():
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(data)

            for k in k_values:
                consistency = 0

                for _ in range(num_iterations):
                    predicted_labels = fit_kmeans(standardized_data, k)

                    if len(np.unique(predicted_labels)) == k:
                        consistency += 1

                consistency_ratio = consistency / num_iterations

                if consistency_ratio < 0.8:
                    sensitive_datasets.append(dataset_abbr)
                    break   

        return sensitive_datasets

    sensitive_datasets = analyze_initialization_sensitivity(answers["1A: datasets"], answers["1B: fit_kmeans"])

    answers["1D: datasets sensitive to initialization"] = sensitive_datasets

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
