import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def hierarchical_clustering(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform hierarchical clustering on a dataset and return the mean coordinates of the clusters.

    Parameters
    ----------
    data : ndarray
        An n x d array representing n data points in d dimensions.
    k : int
        The number of clusters to obtain.

    Returns
    -------
    labels : ndarray
        A 1-dimensional ndarray representing the cluster labels of each data point.
    clusters : ndarray
        A k x d ndarray representing the mean coordinates of the data points for each cluster.
    """

    # Compute the pairwise distances between all pairs of data points
    distances = pdist(data)

    # Perform hierarchical clustering on the pairwise distances
    clusters = linkage(distances, method='ward')

    # Extract the labels of each data point based on the number of clusters
    labels = fcluster(clusters, k, criterion='maxclust')

    # Extract the coordinates of the data points for each cluster
    coords = [data[np.where(labels == i)[0], :] for i in range(1, k+1)]

    # Calculate the mean of each cluster
    means = []
    for cluster in coords:
        mean_x = np.mean(cluster[:, 0])
        mean_y = np.mean(cluster[:, 1])
        mean_z = 1 - (mean_x + mean_y)
        means.append(np.array([mean_x, mean_y, mean_z]))

    # Stack the mean coordinates of each cluster
    clusters = np.stack(means)

    return labels, clusters


def replace_with_clusters(data: np.ndarray, labels: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Replace the points in a data array with the corresponding cluster values based on the labels.

    Parameters
    ----------
    data : ndarray
        An n x d array representing n data points in d dimensions.
    labels : ndarray
        An n x 1 array representing the cluster labels of the data points.
    clusters : ndarray
        A k x 1 array representing the cluster values.

    Returns
    -------
    new_data : ndarray
        An n x d array representing the data points with cluster values.

    """

    # Create an empty array to store the new values
    new_data = np.empty_like(data)

    # Iterate through each point in the data array
    for i in range(data.shape[0]):
        # Replace the point with the corresponding cluster value based on the label
        new_data[i, :] = clusters[labels[i]-1]

    return new_data

