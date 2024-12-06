import numpy as np
from collections import Counter
from ._node import Node
from ._decision_tree_base import DecisionTreeBase
from mlkit.metrics import classification_score
from typing import Union, Optional, List, Literal

class DecisionTreeClassifier(DecisionTreeBase):
    """
    Decision Tree Classifier implementation that inherits from DecisionTreeBase.
    Uses either Gini impurity or log loss as the criterion to build the decision tree.
    """

    def __init__(self,
                 criterion: Optional[Literal["gini", "log_loss"]] = "gini",
                 splitter: Optional[Literal["best", "random"]] = "best",
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[Union[int, float]] = 2,
                 min_samples_leaf: Optional[Union[int, float]] = 1,
                 max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]] = None,
                 random_state: Optional[int] = 42,
                 max_leaf_nodes: Optional[int] = None,
                 min_info_gain: Optional[float] = 0.0):
        """
        Initialize the DecisionTreeClassifier.

        Args:
            criterion (str, optional): The function to measure the quality of a split. Supported values are "gini" (default) and "log_loss".
            splitter (str, optional): The strategy used to split at each node. Supported values are "best" (default) and "random".
            max_depth (int, optional): The maximum depth of the tree. Default is None, which means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            min_samples_split (int or float, optional): The minimum number of samples required to split an internal node. Default is 2.
            min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node. Default is 1.
            max_features (int, float, or str, optional): The number of features to consider when looking for the best split. Default is None, which means all features are considered.
            random_state (int, optional): The seed used by the random number generator. Default is 42.
            max_leaf_nodes (int, optional): The maximum number of leaf nodes in the tree. Default is None.
            min_info_gain (float, optional): The minimum information gain required to split a node. Default is 0.0.
        """
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_info_gain=min_info_gain
        )

   

    def _traverse_tree(self, sample: np.ndarray, node: Node, return_proba=False):
        """
        Traverse the decision tree to predict the class or probabilities for a sample.

        Args:
            sample (np.ndarray): The input data sample to classify.
            node (Node): The current node in the decision tree.
            return_proba (bool, optional): Whether to return class probabilities instead of the predicted class. Default is False.

        Returns:
            np.ndarray or str: Predicted class probabilities as a NumPy array if return_proba is True,
                            otherwise the most common class label.
        """
        while node.best_split:
            threshold = node.best_split_threshold
            feature_value = sample[node.best_split_feature]

            if ((not isinstance(threshold, str) and feature_value > threshold) or
                    (isinstance(threshold, str) and feature_value == threshold)):
                node = node.best_split[0]
            else:
                node = node.best_split[1]

        if return_proba:
            # Use the labels from the parent node for consistent class ordering
            unique_classes = sorted(set(self.parent.y))
            class_counts = Counter(node.y)

            # Create a probability array matching the order of unique classes in the parent
            total_samples = len(node.y)
            proba = np.zeros(len(unique_classes))
            for i, cls in enumerate(unique_classes):
                proba[i] = class_counts[cls] / total_samples if cls in class_counts else 0.0
            return proba

        # Return the most common class label
        return Counter(node.y).most_common(1)[0][0]


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for a batch of input samples.

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix containing samples to predict.

        Returns:
        --------
        np.ndarray
            Array of predicted class labels for each sample in X.
        """
        return np.array([self._traverse_tree(sample, self.parent) for sample in X])


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input samples.

        Args:
            X (np.ndarray): The input data samples.

        Returns:
            np.ndarray: An array of predicted class probabilities for each sample.
        """
        return np.array([self._traverse_tree(sample, self.parent, True) for sample in X])


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the classification score of the model on the given data.

        Args:
            X (np.ndarray): The input data samples.
            y (np.ndarray): The true labels for the input samples.

        Returns:
            float: The classification score of the model.
        """
        preds = self.predict(X)
        return classification_score(y, preds)  # Use external metric function to compute the score
