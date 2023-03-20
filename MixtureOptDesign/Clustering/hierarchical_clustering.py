import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from MixtureOptDesign.MNL.mnl_utils import get_i_optimality_mnl
from sklearn.cluster import AgglomerativeClustering



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


class Cluster:
    def __init__(self, design:np.ndarray):
        x = design[0].flatten()
        y = design[1].flatten()
        z = design[2].flatten()
        # Stack the x, y, and z arrays vertically
        self.design_ = design
        data = np.vstack((x, y, z)).T
        self.data = data
        self.labels = None
        self.clusters = None

    def fit(self, k:int)-> np.ndarray:
        pass

    def design(self) -> np.ndarray:
        # Check if clusters have been computed
        if self.clusters is None:
            raise ValueError("No clusters found. Call 'fit' method first.")

        # Use the replace_with_clusters function to assign each data point to its corresponding cluster
        return replace_with_clusters(self.data, self.labels, self.clusters)
    
    def get_elbow_curve(self,beta:np.ndarray,order:int):
        """
        Plots the elbow curve for the given clustering algorithm between the start and end values of k.

        Parameters
        ----------
        beta : numpy.ndarray of shape (p,)
                A 1-dimensional array of p numbers of beta coefficients for the model.
            
        order : int
                The maximum order of interactions to include in the model. Must be 1, 2 or 3.
            

        Returns
        -------
        plot

        Notes
        -----
        The function computes the I optimality for each value of k between start (number of parameters) and end (unique point) and plots it.
        """
        # Compute the I-optimality criterion for each value of k
        i_opt_values = []
        
        # Set the maximum number of clusters
        max_clusters = self.get_unique_rows(self.data)

        for k in range(beta.size + 1, max_clusters + 1):
            # Fit the clustering algorithm for the given value of k
            self.fit(k)
            
            # Replace the data points with cluster values
            replaced_data = self.design()
            cluster_design = replaced_data.T.reshape(self.design_.shape)
            
            # Calculate the I-optimality criterion for the replaced values
            i_opt = get_i_optimality_mnl(cluster_design, order, beta)

            # Store the I-optimality criterion value
            i_opt_values.append(i_opt)


        # Plot the elbow curve
        plt.plot(range(beta.size + 1, max_clusters + 1), i_opt_values, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('I-optimality')
        plt.title('Elbow curve')
        plt.show()
    

    def get_unique_rows(self,arr:np.ndarray, tolerance=1e-9)->int:
        """
        Get the unique rows of a 2D numpy array based on a specified tolerance level for element-wise equality comparisons.

        Parameters
        ----------
        arr : numpy.ndarray
            The 2D numpy array to get the unique rows of.
        tolerance : float, optional
            The tolerance level for element-wise equality comparisons. Default is 1e-9.

        Returns
        -------
        int
            The length of the unique rows of the input array.

        """
        # Use numpy.isclose to compare the rows with a tolerance level
        close_arr = np.isclose(arr[:, None, :], arr, rtol=tolerance, atol=tolerance).all(axis=2)

        # Use numpy.unique to select the unique rows
        unique_rows_idx = np.unique(close_arr, axis=0, return_index=True)[1]
        unique_rows_idx.sort()
        unique_rows = arr[unique_rows_idx]

        return unique_rows.shape[0]



class HierarchicalCluster(Cluster):
    def fit(self, k:int) -> np.ndarray:
        # Compute pairwise distances between all pairs of data points
        distances = pdist(self.data)

        # Perform hierarchical clustering on the pairwise distances using the 'ward' method
        self.clusters = linkage(distances, method='ward')

        # Extract the labels of each data point based on the number of clusters
        self.labels = fcluster(self.clusters, k, criterion='maxclust')

        # Extract the coordinates of the data points for each cluster
        coords = [self.data[np.where(self.labels == i)[0], :] for i in range(1, k+1)]

        # Calculate the mean of each cluster
        means = []
        for cluster in coords:
            # Compute the mean of each dimension separately and append to means list
            if len(cluster) == 0:
                means.append(np.zeros(self.data.shape[1]))
            else:
                means.append(np.mean(cluster, axis=0))
            

        # Stack the mean coordinates of each cluster
        self.clusters = np.stack(means)
        
    def dendogram(self):
        
        dendrogram(self.clusters)
        plt.show()


from sklearn.cluster import KMeans

class KMeansCluster(Cluster):
    def fit(self, k:int) -> np.ndarray:
        # Create a KMeans object with the desired number of clusters
        kmeans = KMeans(n_clusters=k)

        # Fit the KMeans model to your data
        kmeans.fit(self.data)

        # Get the cluster labels and centroids from the KMeans object
        self.labels = kmeans.labels_
        self.clusters = kmeans.cluster_centers_

        # Return the cluster centroids
        return self.clusters



class AgglomerativeCluster(Cluster):
    def fit(self, k:int) -> np.ndarray:
        # Create an AgglomerativeClustering object with the desired number of clusters
        agglomerative = AgglomerativeClustering(n_clusters=k)

        # Fit the AgglomerativeClustering model to your data
        agglomerative.fit(self.data)

        # Get the cluster labels and clusters from the AgglomerativeClustering object
        labels = agglomerative.labels_
        clusters = [self.data[labels == label] for label in np.unique(labels)]

        # Calculate the mean coordinates of each cluster ensuring they sum up to one
        means = []
        for cluster in clusters:
            mean_x = np.mean(cluster[:, 0])
            mean_y = np.mean(cluster[:, 1])
            mean_z = 1 - (mean_x + mean_y)
            means.append(np.array([mean_x, mean_y, mean_z]))

        # Stack the mean coordinates of each cluster
        centroids = np.stack(means)

        # Create a new numpy array with the same data ordering as the original but with cluster assignments
        ordered_data = np.zeros_like(self.data)
        for label in np.unique(labels):
            ordered_data[labels == label] = centroids[label]

        # Return the ordered data
        return ordered_data


    
    
