import numpy as np
from scipy.ndimage import shift
import math
from mlkit.metrics.classification import multi_log_loss
from typing import Optional, Literal, Union, List

class Node:

    """
    Represents a node in the decision tree. Handles feature splitting and impurity calculation.
    """

    def __init__(self, X, y,
                 criterion: Optional[Literal["gini", "log_loss"]] = "gini", 
                 splitter: Optional[Literal["best", "random"]] = "best",
                 min_samples_split: Optional[Union[int, float]] = 2,
                 min_samples_leaf: Optional[Union[int, float]] = 1,
                 max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]] = None,
                 min_info_gain: Optional[float] = 0.0):
        
        """
        Initializes a Node object for splitting data.

        Args:
            X: Features matrix (2D numpy array).
            y: Target labels (1D numpy array).
            criterion: Impurity criterion ("gini", "log_loss", etc.).
            splitter: Method for selecting thresholds ("best", "random").
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples required in each leaf.
            max_features: Maximum features to consider for splitting.
            min_info_gain: Minimum information gain required for a split.
        """

        self.X = X
        self.y = y
        assert len(X) == len(y), "Lengths of features and labels don't match."
        self.n_samples = len(y)
        self.n_feats = len(X[0]) if self.n_samples > 0 else 0

        self.split_criteria = {
            "criterion": criterion,
            "splitter": splitter,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "min_info_gain": min_info_gain
        }
        self.split_criteria_ = self.split_criteria.copy()
        self._validate_split_criteria()
        self._compute_impurity()

        self.best_split = None
        self.best_split_feature = None
        self.best_split_threshold = None
        self.best_split_impurity = float('inf')
        self.best_split_info_gain = 0



    def _validate_split_criteria(self):

        """
        Validates and adjusts split criteria for the node.
        """

        if isinstance(self.split_criteria["min_samples_split"], float):
            self.split_criteria["min_samples_split"] = math.ceil(
                self.split_criteria["min_samples_split"] * self.n_samples
            )
        if isinstance(self.split_criteria["min_samples_leaf"], float):
            self.split_criteria["min_samples_leaf"] = math.ceil(
                self.split_criteria["min_samples_leaf"] * self.n_samples
            )
        if isinstance(self.split_criteria["max_features"], float):
            self.split_criteria["max_features"] = math.ceil(
                self.split_criteria["max_features"] * self.n_feats
            )
        elif self.split_criteria["max_features"] == "sqrt":
            self.split_criteria["max_features"] = int(np.sqrt(self.n_feats))
        elif self.split_criteria["max_features"] == "log2":
            self.split_criteria["max_features"] = int(np.log2(self.n_feats))

        assert (not self.split_criteria["max_features"] or 
                0 < self.split_criteria["max_features"] <= self.n_feats), (
            "max_features param must not exceed total number of features"
        )
        assert self.split_criteria["min_samples_split"] >= 2, (
            "min_samples_split must be greater than or equal to 2"
        )
        assert self.split_criteria["min_samples_leaf"] >= 1, (
            "min_samples_leaf must be greater than or equal to 1"
        )
        assert self.split_criteria["min_info_gain"] >= 0, (
            "min_info_gain must be a positive integer"
        )



    def _compute_impurity(self):

        """
        Computes the impurity of the current node based on the selected criterion.
        """

        if self.split_criteria["criterion"] == "gini":
            imp = 1
            for value in np.unique(self.y):
                occurrence = np.sum(self.y == value)
                imp -= (occurrence / len(self.y)) ** 2

        elif self.split_criteria["criterion"] == "log_loss":
            class_counts = np.bincount(self.y)  # Count occurrences of each class
            probabilities = class_counts / self.n_samples  # Compute probabilities for each class
            imp = multi_log_loss(self.y, probabilities) # multi_log_loss impurity

        elif self.split_criteria["criterion"] == "mean_squared_error":
            pred_val = np.mean(self.y)
            imp = np.mean((pred_val - self.y)**2)
        elif self.split_criteria["criterion"] == "mean_absolute_error":
            pred_val = np.mean(self.y)
            imp = np.mean(np.abs(pred_val-self.y))

        else:
            raise ValueError(f"Unsupported criterion: {self.split_criteria['criterion']}")

        self.node_impurity = imp


    def split_by(self, feature, threshold):

        """
        Splits the node into two child nodes based on a feature and threshold.

        Args:
            feature: The feature index to split on.
            threshold: The threshold value for splitting.

        Returns:
            Two child Node objects resulting from the split.
        """

        if type(threshold) != str:
            mask = (self.X[:, feature]) > threshold
        else:
            mask = (self.X[:, feature]) == threshold
        X_above, y_above = self.X[mask], self.y[mask]
        X_below, y_below = self.X[~mask], self.y[~mask]
        node_1 = Node(X_above, y_above, **self.split_criteria_)
        node_2 = Node(X_below, y_below, **self.split_criteria_)
        return node_1, node_2



    def split(self):
        """
        Performs the best possible split based on the feature and threshold.

        Returns:
            A tuple containing two child nodes resulting from the split.
        """
        # Check if we have enough samples to split
        if self.n_samples < self.split_criteria["min_samples_split"]:
            return None
        
        # Determine features to consider
        feats_idx = list(range(len(self.X[0])))
        if self.split_criteria["max_features"]:
            max_features = min(self.split_criteria["max_features"], self.n_feats)
            feats_idx = np.random.choice(feats_idx, max_features, replace=False)

        best_split = None
        best_split_impurity = float("inf")
        best_split_feature = None
        best_split_threshold = None
        best_split_info_gain = 0

        # Iterate through features
        for feature in feats_idx:
            # Determine possible thresholds
            if not isinstance(self.X[0, feature], str):
                if self.split_criteria["splitter"] == "best":
                    # Midpoints between unique values
                    unique_vals = np.unique(self.X[:, feature])
                    possible_thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
                else:
                    # Random threshold
                    unique_vals = np.unique(self.X[:, feature])
                    possible_thresholds = [np.random.choice(unique_vals)]
            else:
                # Categorical features
                possible_thresholds = np.unique(self.X[:, feature])
                if self.split_criteria["splitter"] == "random":
                    possible_thresholds = [np.random.choice(possible_thresholds)]

            for threshold in possible_thresholds:
                # Split the node
                node_1, node_2 = self.split_by(feature, threshold)

                # Ensure we don't create empty nodes
                if len(node_1.y) == 0 or len(node_2.y) == 0:
                    continue
                # Skip if child nodes are too small
                if (node_1.n_samples < self.split_criteria["min_samples_leaf"] or 
                    node_2.n_samples < self.split_criteria["min_samples_leaf"]):
                    continue

                # Calculate weighted impurity
                split_impurity = (
                    (node_1.node_impurity * node_1.n_samples + 
                     node_2.node_impurity * node_2.n_samples) / 
                    self.n_samples
                )

                # Calculate info gain
                info_gain = self.node_impurity - split_impurity
                # Update best split if better
                if split_impurity < best_split_impurity and info_gain > self.split_criteria["min_info_gain"]:
                    best_split_impurity = split_impurity
                    best_split = [node_1, node_2]
                    best_split_feature = feature
                    best_split_threshold = threshold
                    best_split_info_gain = info_gain
        
                # print(best_split, best_split_impurity, best_split_feature, best_split_threshold, best_split_info_gain)

        # If a good split is found, update node and return
        if best_split:
            self.best_split = best_split
            self.best_split_feature = best_split_feature
            self.best_split_threshold = best_split_threshold
            self.best_split_impurity = best_split_impurity
            self.best_split_info_gain = best_split_info_gain
            return self.best_split
    
        return None