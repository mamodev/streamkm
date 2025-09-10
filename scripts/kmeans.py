# use scypy kmeans 
import argparse
import numpy as np
import os

from parser import read_dataset,read_data_from_file


argparser = argparse.ArgumentParser(description='Run k-means clustering on a dataset.')
argparser.add_argument('k', type=int, help='Number of clusters.')
argparser.add_argument('centers_file', type=str, help='Path to the file containing points used to calculate centers.')
argparser.add_argument('ds_folder_or_datafile_path', type=str, help='Path to the dataset folder or data file. if folder provided defaults to folder/data.bin')
argparser.add_argument('--ds_labels_file', type=str, required=False, help='Required if ds_folder_or_datafile_path is a data file. With folder defaults to folder/labels.bin')


args = argparser.parse_args()

assert args.ds_labels_file or os.path.isdir(args.ds_folder_or_datafile_path), "If dataset path is a file, labels file must be provided"

datafile_path = args.ds_folder_or_datafile_path if os.path.isfile(args.ds_folder_or_datafile_path) else os.path.join(args.ds_folder_or_datafile_path, 'data.bin')
labels_file = args.ds_labels_file if args.ds_labels_file else os.path.join(args.ds_folder_or_datafile_path, 'labels.bin')
ds_points, ds_labels = read_dataset(datafile_path, labels_file)
centroids = read_data_from_file(args.centers_file)

# print some info
print(f"Dataset shape: {ds_points.shape}")
print(f"Dataset labels shape: {ds_labels.shape}")
print(f"Centroids shape: {centroids.shape}")
assert centroids.shape[1] == ds_points.shape[1], "Dimensionality of centroids and dataset points must match"

# check how many unique labels in ds_labels
num_unique_labels = len(np.unique(ds_labels))
print(f"Number of unique labels in dataset: {num_unique_labels}")

from contingency_matrix import ContingencyMatrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min

# cluster the centroids themselves
kmeans_on_centroids = KMeans(
    n_clusters=args.k,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans_on_centroids.fit(centroids)


# these are the k "super-centers"
cluster_centers = kmeans_on_centroids.cluster_centers_

# print centers
print("Cluster centers:")
for i, center in enumerate(cluster_centers):
    print(f"Center {i}: {center}")

final_labels = pairwise_distances_argmin_min(ds_points, cluster_centers)[0]

cm = ContingencyMatrix(ds_labels, final_labels)
print("Purity:", cm.purity())
print("ARI:", cm.adjusted_rand_index())
print("NMI:", cm.normalized_mutual_info())
print("FMI:", cm.fowlkes_mallows())


