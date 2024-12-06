import numpy as np
from collections import Counter
from mlkit.preprocessing import MinMaxScaler
from mlkit.metrics import classification_score

class KNeighborsClassifier:
    """
    K-Nearest Neighbors classifier.
    
    Parameters
    ----------
    k : int, optional (default=5)
        Number of neighbors to use.
    metric : str, optional (default="d2")
        Distance metric to use. Supported metrics are:
        - "d2" or "euclidean" for Euclidean distance
        - "d1", "manhattan", or "abs" for Manhattan distance
        - "dinf", "max", or "Chebyshev" for Chebyshev distance
    
    Attributes
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training data.
    y_train : array-like, shape (n_samples,)
        Training labels.
    labels : array-like, shape (n_classes,)
        Unique class labels.
    
    Methods
    -------
    fit(X, y)
        Fit the model using X as training data and y as target values.
    predict(X)
        Predict the class labels for the provided data.
    _get_distance(p1, p2)
        Compute the distance between two points based on the specified metric.
    _get_k_neighbors(center)
        Find the k nearest neighbors of a given point.
    """

    def __init__(self, k=5, metric="d2", scaler=MinMaxScaler()):
        """
        Initialize the KNeighborsClassifier with specified parameters.
        
        Parameters
        ----------
        k : int, optional (default=5)
            Number of neighbors to consider.
        metric : str, optional (default="d2")
            Distance metric to use for distance computation.
        scaler: object, optional (default=MinMaxScaler())
            Scaler transformer used to scale features.
        """
        self.k = k
        self.metric = metric
        self.scaler = scaler
        self.X_train = None
        self.y_train = None
        self.labels = None
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.
        """
        self.scaler = type(self.scaler)()
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = y
        self.labels = np.unique(y)
        self.is_fitted = True
        return self

    def _get_distance(self, p1, p2):
        """
        Compute the distance between two points p1 and p2 based on the specified metric.
        
        Parameters
        ----------
        p1 : array-like, shape (n_features,)
            First data point.
        p2 : array-like, shape (n_features,)
            Second data point.
        
        Returns
        -------
        float
            Computed distance between p1 and p2.
        """
        if self.metric.lower() in ["d2", "euclidean"]:
            return np.sqrt(np.sum((p1 - p2) ** 2))
        elif self.metric.lower() in ["d1", "manhattan", "abs"]:
            return np.sum(np.abs(p1 - p2))
        elif self.metric.lower() in ["dinf", "max", "chebyshev"]:
            return np.max(np.abs(p1 - p2))

    def _get_k_neighbors(self, center):
        """
        Find the k nearest neighbors of a given point in the training data.
        
        Parameters
        ----------
        center : array-like, shape (n_features,)
            The point for which the nearest neighbors are to be found.
        
        Returns
        -------
        list of tuples
            A list of the k nearest neighbors, each represented as a tuple (distance, label).
        """
        distances = [self._get_distance(center, x) for x in self.X_train]
        dist_dict = list(zip(distances, self.y_train))
        dist_dict.sort()
        return dist_dict[:self.k]

    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data for which to predict class labels.
        
        Returns
        -------
        array-like, shape (n_samples,)
            Predicted class labels.
        """
        X = self.scaler.transform(X)
        preds = []
        for element in X:
            neighbors = self._get_k_neighbors(element)
            neighbors_labels = [label for _, label in neighbors]
            pred_label = Counter(neighbors_labels).most_common(1)[0][0]
            preds.append(pred_label)
        return np.array(preds)

    def score(self, X, y):
        """
        Calculate the classification score of the model.

        Parameters:
        X (array-like): Feature data used for prediction.
        y (array-like): True labels corresponding to the feature data.

        Returns:
        float: The classification score of the model.

        Raises:
        AssertionError: If the model is not fitted before calling this method.
        """
        assert self.is_fitted == True, "Please fit the model first."
        y_pred = self.predict(X)
        return classification_score(y, y_pred)