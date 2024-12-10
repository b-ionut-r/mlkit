import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Union, Literal, Self, Callable

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two points.
    
    Parameters
    ----------
    p1 : np.ndarray
        The first point.
    p2 : np.ndarray
        The second point.

    Returns
    -------
    float
        The Euclidean distance between p1 and p2.
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))

def norm(x: np.ndarray) -> float:
    """
    Compute the Euclidean norm (magnitude) of a vector.
    
    Parameters
    ----------
    x : np.ndarray
        The vector for which the norm is computed.

    Returns
    -------
    float
        The norm of the vector x.
    """
    return np.sqrt(np.sum(x ** 2))


class KMeans:
    """
    KMeans clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form and the number of centroids to generate.
    init : {'k-means++', 'random'} or callable or ndarray, default='k-means++'
        Method for initialization of centroids.
    n_init : {'auto'} or int, default='auto'
        Number of times the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    verbose : int, default=0
        Verbosity mode.
    random_state : int, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    centroids_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. Will be None if the model has not been fitted yet.
    inertia_ : float
        Inertia (sum of squared distances to the nearest centroid).
    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point indicating the cluster to which they belong.
    """

    def __init__(self, 
                 n_clusters: int = 8,
                 init: Union[Literal["k-means++", "random"], Callable, np.ndarray] = "k-means++",
                 n_init: Union[Literal["auto"], int] = "auto",
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: int = None,
                 ):
        """
        Initialize the KMeans clustering algorithm.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        np.random.seed(random_state)
        self.seeds_ = None
        self.n_samples_ = None
        self.n_feats_ = None
        self.centroids_ = None
        self.inertia_ = None
        self.centroids_iters_ = []
        self.inertias_iters_ = []
        self.labels_ = None

    def _generate_clusters_from_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Cluster labels for each sample.
        """
        distances = np.array([[distance(x, self.centroids_[k]) for k in range(self.n_clusters)] for x in X])
        labels = np.argmin(distances, axis=-1)
        return labels

    def _get_clusters_means(self, X: np.ndarray) -> np.ndarray:
        """
        Compute new centroids by calculating the mean of each cluster.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            New centroids of shape (n_clusters, n_features).
        """
        new_centroids = np.zeros_like(self.centroids_)
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Fallback for empty clusters
                new_centroids[k] = X[np.random.randint(self.n_samples_)]
        return new_centroids

    def _get_inertia(self, X: np.ndarray) -> float:
        """
        Calculate the inertia (sum of squared distances to the nearest centroid).
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        float
            The inertia of the clustering.
        """
        inertia = 0
        for k in range(self.n_clusters):
            k_idx = (self.labels_ == k)
            if k_idx.any():
                X_k = X[k_idx]
                inertia += np.sum([distance(x, self.centroids_[k]) ** 2 for x in X_k])
        return inertia

    def fit(self, X: np.ndarray) -> Self:
        """
        Compute KMeans clustering.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        Self
            The fitted KMeans instance.
        """
        self.n_samples_, self.n_feats_ = X.shape
        self.centroids_ = np.zeros((self.n_clusters, self.n_feats_))
        self.labels_ = np.zeros(self.n_samples_)

        if self.n_init == "auto":
            if self.init == "random" or callable(self.init):
                self.n_init = 100
            else:
                self.n_init = 10
        self.seeds_ = np.random.randint(0, 1000, self.n_init)

        iterable = tqdm(range(self.n_init)) if self.verbose else range(self.n_init)
        for i in iterable:
            np.random.seed(self.seeds_[i])
            if self.init == "k-means++":
                self.centroids_ = np.random.uniform(X.min(axis=0), X.max(axis=0), 
                                                    (self.n_clusters, self.n_feats_))
            elif self.init == "random":
                self.centroids_ = X[np.random.choice(self.n_samples_, self.n_clusters, replace=False)]
            elif isinstance(self.init, np.ndarray):
                assert self.init.shape == (self.n_clusters, self.n_feats_), \
                 "Inital centers array must be of shape: (n_clusters, n_features)."
                self.centroids_ = self.init
            elif isinstance(self.init, callable):
                self.centroids_ = self.init(X, self.n_clusters, self.seeds_[i])
                assert self.centroids_.shape == (self.n_clusters, self.n_feats_), \
                 "Centers array returned by callable init must be of shape: (n_clusters, n_features)."

            err = 1
            iter = 0
            while iter < self.max_iter and norm(err) > self.tol:
                self.labels_ = self._generate_clusters_from_centroids(X)
                means = self._get_clusters_means(X)
                err = self.centroids_ - means
                self.centroids_ = means
                iter += 1

            self.centroids_iters_.append(self.centroids_)
            self.inertias_iters_.append(self._get_inertia(X))

        self.inertia_ = np.min(self.inertias_iters_)
        self.centroids_ = self.centroids_iters_[np.argmin(self.inertias_iters_)]
        self.labels_ = self._generate_clusters_from_centroids(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted cluster labels for each sample.
        """
        assert self.centroids_ is not None, "Please fit the model first."
        return self._generate_clusters_from_centroids(X)

    def test_n_clusters(self, X: np.ndarray, upto=10):
        """
        Test different values for the number of clusters and plot inertia for each.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        upto : int, default=10
            The maximum number of clusters to test.

        Returns
        -------
        None
            Plots the inertia for each number of clusters.
        """
        inertias = []
        iterable = tqdm(range(1, upto + 1)) if self.verbose else range(1, upto + 1)
        for i in iterable:
            model = KMeans(n_clusters=i, 
                           init=self.init,
                           n_init=self.n_init,
                           max_iter=self.max_iter,
                           tol=self.tol,
                           verbose=0,
                           random_state=self.random_state)
            inertias.append(model.fit(X).inertia_)
        plt.plot(range(1, upto + 1), inertias)
        plt.show()
