import numpy as np
from scipy.ndimage import shift
from collections import Counter
from mlkit.metrics import classification_score




class DecisionTreeClassifier:

    """
    A simple implementation of a Decision Tree Classifier.
    Attributes:
    -----------
    parent : Node
        The root node of the decision tree.
    levels : list
        A list of levels in the decision tree.
    Methods:
    --------
    __init__():
        Initializes the DecisionTreeClassifier with default values.
    fit(X, y):
        Fits the decision tree classifier to the provided data.
    _traverse_tree(sample, node):
        Traverses the decision tree to make a prediction for a single sample.
    predict(X):
        Predicts the class labels for the provided data.
    score(X, y):
        Returns classification evaluation metrics on the given test data and labels.
    """

    def __init__(self):

        self.parent = None
        self.levels = []

    def fit(self, X, y):

        self.parent = Node(X, y)
        level = Level(self.parent)
        self.levels.append(level)

        next_level = level.split()
        while next_level:
            self.levels.append(next_level)
            next_level = next_level.split()

    
    def _traverse_tree(self, sample, node):
        while node.best_split:
            if sample[node.best_split_feature] > node.best_split_threshold:
                node = node.best_split[0]
            else:
                node = node.best_split[1]
        return Counter(node.y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(sample, self.parent) for sample in X])
    
    def score(self, X, y):
        preds = self.predict(X)
        return classification_score(y, preds)






class Node:

    """
    A class used to represent a Node in a Decision Tree.
    Attributes
    ----------
    X : ndarray
        The feature matrix for the node.
    y : ndarray
        The target values for the node.
    n_samples : int
        The number of samples in the node.
    node_impurity : float
        The Gini impurity of the node.
    best_split : list
        The best split nodes (left and right) based on impurity.
    best_split_feature : int
        The feature index used for the best split.
    best_split_threshold : float
        The threshold value used for the best split.
    best_split_impurity : float
        The impurity value of the best split.
    best_split_info_gain : float
        The information gain from the best split.
    Methods
    -------
    __init__(X, y)
        Initializes the Node with feature matrix X and target values y.
    compute_gini_impurity()
        Computes the Gini impurity for the node.
    split_by(feature, threshold)
        Splits the node into two child nodes based on the given feature and threshold.
    split()
        Finds the best split for the node by evaluating all possible splits.
    """

    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.n_samples = len(y)
        self.compute_gini_impurity()
        self.best_split = None
        self.best_split_feature = None
        self.best_split_threshold = None
        self.best_split_impurity = 1
        self.best_split_info_gain = 0
    


    def compute_gini_impurity(self):
        imp = 1
        for value in np.unique(self.y):
            occurence = np.sum((self.y == value))
            imp -= (occurence / len(self.y)) ** 2
        self.node_impurity = imp


    def split_by(self, feature, threshold): # binary tree
        mask = (self.X[:, feature]) > threshold
        X_above, y_above = self.X[mask], self.y[mask]
        X_below, y_below = self.X[~mask], self.y[~mask]
        node_1 = Node(X_above, y_above) # > threshold
        node_2 = Node(X_below, y_below) # <= threshold
        return node_1, node_2
    

    def split(self):
        for feature in range(len(self.X[0])):
            possible_thresholds = ((shift(np.unique(self.X[:, feature]), 1) + np.unique(self.X[:, feature])) / 2)[1:]
            for threshold in possible_thresholds:
                node_1, node_2 = self.split_by(feature, threshold)
                split_impurity = (node_1.node_impurity * node_1.n_samples +
                                  node_2.node_impurity * node_2.n_samples) / self.n_samples
                if split_impurity < self.best_split_impurity:
                    self.best_split_impurity = split_impurity
                    self.best_split = [node_1, node_2]
                    self.best_split_feature = feature
                    self.best_split_threshold = threshold
                    self.best_split_info_gain = self.node_impurity - self.best_split_impurity
        return self.best_split




class Level:
    """
    A class used to represent a Level in a decision tree.
    Attributes
    ----------
    nodes : list
        A list of nodes at the current level of the decision tree.
    Methods
    -------
    __init__(*nodes)
        Initializes the Level with the given nodes.
    split()
        Splits the nodes at the current level and returns a new Level with the resulting nodes if any splits are possible.
    """

    def __init__(self, *nodes):
        self.nodes = []
        for node in nodes:
            self.nodes.append(node)

    def split(self):
        new_nodes = []
        for node in self.nodes:
            splits = node.split()
            if splits and node.best_split_info_gain > 0:
                new_nodes.extend(splits)
        if new_nodes:
            return Level(*new_nodes)
        else:
            # print("Can't split anymore.")
            return None

