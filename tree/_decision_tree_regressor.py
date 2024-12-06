import numpy as np
from ._node import Node
from ._decision_tree_base import DecisionTreeBase
from mlkit.metrics import regression_score
from typing import Union, Optional, List, Literal

class DecisionTreeRegressor(DecisionTreeBase):
    """
    A Decision Tree Regressor implementation for predicting continuous values.

    Parameters:
    - criterion (str, optional): The function to measure the quality of a split.
      Supported options are "mean_squared_error" for MSE and "mean_absolute_error" for MAE.
      Default is "mean_squared_error".
    - splitter (str, optional): The strategy used to split at each node.
      Supported options are "best" to select the best split and "random" for random splits.
      Default is "best".
    - max_depth (int or None, optional): The maximum depth of the tree. If None, the tree is expanded until leaves are pure.
    - min_samples_split (int or float, optional): The minimum number of samples required to split an internal node.
      If an integer, it represents the number of samples. If a float, it represents the fraction of samples.
      Default is 2.
    - min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node.
      If an integer, it represents the number of samples. If a float, it represents the fraction of samples.
      Default is 1.
    - max_features (int, float, str, or None, optional): The number of features to consider when looking for the best split.
      If an integer, it represents the number of features. If a float, it represents the fraction of features.
      If "sqrt", it considers the square root of the number of features. If "log2", it considers the log base 2 of the number of features.
      Default is None (use all features).
    - random_state (int, optional): The seed used by the random number generator. Default is 42.
    - max_leaf_nodes (int or None, optional): The maximum number of leaf nodes in the tree. If None, there is no limit.
    - min_info_gain (float, optional): The minimum information gain required to split a node. Default is 0.0.

    Attributes:
    - root (Node): The root node of the decision tree.
    """
    
    def __init__(self,
                 criterion: Optional[Literal["mean_squared_error", "mean_absolute_error"]] = "mean_squared_error",
                 splitter: Optional[Literal["best", "random"]] = "best",
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[Union[int, float]] = 2,
                 min_samples_leaf: Optional[Union[int, float]] = 1,
                 max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]] = None,
                 random_state: Optional[int] = 42,
                 max_leaf_nodes: Optional[int] = None,
                 min_info_gain: Optional[float] = 0.0):
        """
        Initialize a Decision Tree Regressor with the specified hyperparameters.
        
        This constructor sets up the model with the parameters used for tree building,
        such as the splitting criteria, tree depth, and other regularization settings.
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


    def _traverse_tree(self, sample: np.ndarray, node: Node) -> float:
        """
        Traverse the decision tree to predict the value for a sample in a regression task.

        Args:
            sample (np.ndarray): The input data sample to predict.
            node (Node): The current node in the decision tree.

        Returns:
            float: The predicted value (mean) for the input sample in the regression task.
        """
        while node.best_split and node:
            threshold = node.best_split_threshold
            feature_value = sample[node.best_split_feature]

            if feature_value > threshold:
                node = node.best_split[0]
            else:
                node = node.best_split[1]

        # Return the mean of the target values in the leaf node
        return np.mean(node.y)
    
    
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


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the R^2 score (coefficient of determination) of the model's predictions.
        
        This method evaluates the accuracy of the regression model by comparing the predicted 
        values with the true target values using a regression score metric.

        Parameters:
        - X (np.ndarray): The input features to predict.
        - y (np.ndarray): The true target values.

        Returns:
        - float: The R^2 score of the model's predictions.
        """
        preds = self.predict(X)
        return regression_score(y, preds)  # Regression score function evaluates the model's performance
