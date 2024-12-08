import numpy as np
import pandas as pd
from mlkit.tree import DecisionTreeRegressor
from mlkit.metrics import regression_score
from typing import Optional, Union, Literal
from tqdm import tqdm
from joblib import Parallel, delayed

class RandomForestRegressor:
    """
    Basic implementation of a Random Forest Regressor.

    This class provides a flexible implementation of the Random Forest algorithm,
    with multiple hyperparameter tuning options and 
    parallel processing capabilities.
       
    """

    def __init__(self, 
                 n_estimators: Optional[int] = 100,
                 criterion: Optional[Union[Literal["mean_squared_error", "mean_absolute_error"]]] = "mean_squared_error",
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = 2,
                 min_samples_leaf: Optional[int] = 1,
                 max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]] = 1.0,
                 max_leaf_nodes: Optional[int] = None,
                 min_info_gain: Optional[float] = 0.0,
                 bootstrap: Optional[bool] = True,
                 max_samples: Optional[Union[int, float]] = None,
                 n_jobs: Optional[int] = 1,
                 random_state: Optional[int] = None,
                 verbose: Optional[int] = 1):
        
        """
        Initialize the Random Forest model with given hyperparameters.

        Args:
            * n_estimators (int): Number of trees in the forest. Defaults to 100.
            * criterion (str): The function to measure the quality of a split. 
                Can be "mean_squared_error" or "mean_absolute_error". Defaults to "mean_squared_error".
            * max_depth (int, optional): Maximum depth of the trees. Defaults to None.
            * min_samples_split (int): Minimum number of samples required to split an internal node. 
                Defaults to 2.
            * min_samples_leaf
              (int): Minimum number of samples required to be at a leaf node. 
                Defaults to 1.
            * max_features (int/float/str): Number of features to consider when looking for the best split.
                Can be an int, float, or "sqrt"/"log2". Defaults to 1.0.
            * max_leaf_nodes (int, optional): Maximum number of leaf nodes. Defaults to None.
            * min_info_gain (float): Minimum information gain required for a split. Defaults to 0.0.
            * bootstrap (bool): Whether bootstrap samples are used when building trees. Defaults to True.
            * max_samples (int/float, optional): Number of samples to draw for each tree. Defaults to None.
            * n_jobs (int): Number of jobs to run in parallel. Defaults to 1.
            * random_state (int, optional): Seed for random number generation. Defaults to None.
            * verbose (int): Verbosity level for progress tracking. Defaults to 1.
        """
            
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_info_gain = min_info_gain 
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
        self.verbose = verbose

        self.trees = []
        self.n_features = None


    def _validate_input(self, X, y=None):
        """
        Validate the input data for consistency and numeric requirements.

        Checks that:
        - X is a numpy array
        - All elements in X are numeric
        - If y is provided, it matches X in length and is a numpy array

        Args:
            X (np.ndarray): Input feature matrix
            y (np.ndarray, optional): Input label vector

        Raises:
            AssertionError: If input validation fails
            ValueError: If non-numeric data is detected
        """
        assert isinstance(X, np.ndarray), "X must be a np.ndarray"
        try:
            pd.to_numeric(X.flatten())
        except ValueError as e:
            raise ValueError("All columns in X must be numeric.") from e
        if y is not None:
            assert isinstance(y, np.ndarray), "y must be a np.ndarray"
            assert len(X) == len(y), "Sizes of inputs and labels don't match"
            try:
                pd.to_numeric(y.flatten())
            except ValueError as e:
                raise ValueError("y must be numeric.") from e



    def fit(self, X, y):
        """
        Fit the random forest regressor to the training data.

        Builds multiple decision trees using bootstrapped samples and 
        pasrallel processing.

        Args:
            X (np.ndarray): Training feature matrix
            y (np.ndarray): Training label vector
        """
        self._validate_input(X, y)

        n_samples = len(X)
        self.n_features = len(X[0])

        iterator = (tqdm(range(self.n_estimators)) if self.verbose
                    else range(self.n_estimators))

        def for_loop(i):
            """
            Create and fit a single decision tree.

            Args:
                i (int): Tree index (used for tracking)

            Returns:
                DecisionTreeRegressor: Fitted decision tree
            """

            np.random.seed(self.random_state + i)
            no_samples = self.max_samples if self.max_samples else n_samples
            indices = np.random.choice(range(n_samples), size=no_samples, replace=self.bootstrap)
          

            tree = DecisionTreeRegressor(criterion=self.criterion,
                                          splitter="best",
                                          max_depth=self.max_depth, 
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf,
                                          max_features=self.max_features,
                                          random_state=self.random_state + i,
                                          max_leaf_nodes=self.max_leaf_nodes,
                                          min_info_gain=self.min_info_gain)
            
            tree.fit(X[indices], y[indices])
            return tree
        
        trees = Parallel(n_jobs=self.n_jobs)(delayed(for_loop)(i) for i in iterator)
        self.trees.extend(trees)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression for input samples.

        Aggregates predictions from all trees using mean.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted targets of shape (n_samples,)

        Raises:
            AssertionError: If model has not been fitted before prediction.
        """
        assert self.trees, "Model must be fitted before prediction"
        self._validate_input(X)
        
        all_preds = []
        for tree in self.trees:
            all_preds.append(tree.predict(X))
            
        return np.mean(all_preds, axis=0)


   
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates regression evaluation metrics on input data.

        Args:
            X (np.ndarray): Test feature matrix of shape (n_samples, n_features)
            y (np.ndarray): True labels of shape (n_samples,)

        Returns:
            float: Regression metrics (mae, mse, rmse, r2).

        Raises:
            AssertionError: If model has not been fitted before scoring
        """
        assert self.trees, "Model must be fitted before scoring"
        self._validate_input(X, y)
        
        y_preds = self.predict(X)
        return regression_score(y, y_preds)
    

    def get_feature_importance(self):
        """
        Calculate feature importance based on how often each feature is used for splitting.

        Returns:
            dict: A dictionary with feature indices as keys and normalized importance values as values.
        """
        assert self.trees, "Please fit the model first."
        
        importances = {k: 0 for k in range(self.n_features)}
        total_splits = 0
        
        for tree in self.trees:
            for node in tree.nodes:
                if node.best_split_feature:
                    importances[node.best_split_feature] += 1
                    total_splits += 1
        
        # Normalize importances by total splits
        importances = {k: v / total_splits for k, v in importances.items()}
        return importances
    

    def plot_tree(self, i, feature_names = None):
        """
        Plot a decision tree from the ensemble.

        Args:
            i (int): Index of the tree to plot.
            feature_names (list): List of feature names for visualization.
        """
        self.trees[i].plot_tree(feature_names)


