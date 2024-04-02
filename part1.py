import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn import datasets
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

def fit_kmeans(data, n_clusters):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    kmeans = cluster.KMeans(n_clusters=n_clusters, init='random')
    kmeans.fit(standardized_data)

    predicted_labels = kmeans.labels_
    return predicted_labels  # Make sure to return the labels



def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """
    

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    random_state=42;

    # Directly call dataset functions from sklearn.datasets
    nc = datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state)
    nm = datasets.make_moons(n_samples=100, noise=.05, random_state=random_state)
    bvv = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    add = datasets.make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    transformed_data = np.dot(add[0], transformation)  # Apply transformation to the data

    # Create a new variable for the transformed dataset
    add_transformed =(transformed_data, add[1])

    b = datasets.make_blobs(n_samples=100, random_state=random_state)


    
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["1A: datasets"] = {'nc': nc, 'nm': nm, 'bvv': bvv, 'add': add_transformed,'b': b}

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    def create_cluster_plots(datasets, fit_kmeans):
        k_values = [2, 3, 5, 10]
        fig, axs = plt.subplots(len(k_values), len(datasets), figsize=(20, 15))
    
    # Initialize cluster successes and failures dictionaries
        cluster_successes = {}
        cluster_failures = []

        for k_index, k in enumerate(k_values):
            for dataset_index, (dataset_abbr, (data, _)) in enumerate(datasets.items()):
                predicted_labels = fit_kmeans(data, k)
            
            # Here you could analyze predicted_labels to populate cluster_successes and cluster_failures
                if dataset_abbr not in cluster_successes:
                    cluster_successes[dataset_abbr] = []
                cluster_successes[dataset_abbr].append(k)
            
                ax = axs[k_index, dataset_index]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
                ax.set_title(f'{dataset_abbr}, k={k}')
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig('cluster_plots.pdf')
        plt.close()
        
        return cluster_successes, cluster_failures
    cluster_successes, cluster_failures = create_cluster_plots(answers["1A: datasets"], answers["1B: fit_kmeans"])

    dct= answers["1C: cluster successes"] = {"bvv":[2, 3, 5, 10], "add":[2,3,5,10], "b":[2,3,5,10]}
    dct = answers["1C: cluster failures"] = {"nc":[2,3,5,10],"nm":[2,3,5,10]}
    

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
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

    dct = answers["1D: datasets sensitive to initialization"] = sensitive_datasets

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
