
import numpy as np
class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            # assign new clustering: create a ( N x (n_clusters) matrix, where element ij represents distance of i-th object to j-th cluster
            # to assign a proper cluster, argmin
            new_clustering = np.argmin(self.euclidean_distance(X, self.centroids), axis=1)
            # check if the difference of the new clustering and the old one is 0 (convergence)
            if np.count_nonzero(new_clustering - clustering) == 0 :
                break
            clustering = new_clustering
            self.update_centroids(clustering, X)
            iteration += 1
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        # for each cluster
        for i in range(self.n_clusters):
            # see objects that belong to current cluster, average the result
            cluster_instances = X[ clustering == i ,:]

            # think about this if statement.
            # I added it here because some clusters can be empty, causing errors
            if cluster_instances.shape[0] > 0:
                self.centroids[i] = np.mean(cluster_instances, axis=0)

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """

        n_rows = X.shape[0]
        n_cols = X.shape[1]
        if self.init == 'random':
            indices = np.random.choice(n_rows, size=self.n_clusters ,replace=False)
            self.centroids = X[indices]
        elif self.init == 'kmeans++':
            # choose the first centroid, add it to centroids list
            self.centroids = np.zeros((1, n_cols))
            self.centroids = X[np.random.choice(n_rows)]

            for i in range( self.n_clusters - 1 ):
                # compute the distance from existing centroids to everydatapoint, take the smallest distance
                distances = np.min( self.euclidean_distance(X, self.centroids), axis=1)
                # normalize the distance vectorm, so they add up to one, and could represent a probability
                prob = distances / (np.sum(distances))
                # choose the next centroid, apped it to centroids
                new_centroid = X[np.random.choice(n_rows, p=prob)]
                self.centroids = np.vstack((self.centroids, new_centroid))

            pass
        else:
            raise ValueError('Centroid initialization method should either be "random" or "kmeans++"')

    # for X1 (MxD) and X2 (N * D), dist (M * N)
    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the Euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # For broadcasting purposes, need to add dimensions to X1, X2
        X1 = np.expand_dims(X1, axis=2)
        X2 = np.expand_dims(X2.T, axis=0)
        # First, create a 3D matrix (X1 objects, features, X2 objects), where all the differences between X1, X2 are stored
        # square the differences, add the up along the axis representding features (axis = 1), then take the square root
        dist = np.sqrt(np.sum((X2 - X1) ** 2, axis=1))

        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        dist_table = self.euclidean_distance(X, self.centroids)
        # for every datapoint, compute a
        a = dist_table[np.arange(dist_table.shape[0]), clustering]
        # for every datapoint, compute b
        # https://stackoverflow.com/questions/22546180/find-nth-smallest-element-in-numpy-array
        b = np.partition(dist_table, 1, axis=1)[:, 1]
        diff = b - a
        silhouette_vector = diff / np.max( np.vstack([a, b]).T, axis=1 )
        return np.mean(silhouette_vector)