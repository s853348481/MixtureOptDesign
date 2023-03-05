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
    means : ndarray
        A 3 x k x 1 ndarray representing the mean coordinates of the data points for each cluster.

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
        means.append(np.array([mean_x, mean_y, mean_z]).reshape(3, 1))

    # Stack the mean coordinates of each cluster
    means = np.stack(means, axis=1)

    return means
