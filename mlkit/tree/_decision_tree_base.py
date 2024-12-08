import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from mlkit.metrics import classification_score
from ._node import Node
from ._level import Level
from typing import Union, Optional, Literal, List
import warnings


class DecisionTreeBase:
    """
    Base implementation of a Decision Tree classifier.

    Parameters:
    -----------
    criterion : str, optional, default="gini"
        The function to measure the quality of a split. Supported criteria are "gini" and "entropy".
    
    splitter : {'best', 'random'}, optional, default="best"
        Strategy used to choose the split at each node. 
        'best' chooses the best split, 'random' chooses the best random split.

    max_depth : int, optional, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

    min_samples_split : int or float, optional, default=2
        The minimum number of samples required to split an internal node.
        If int, it is the minimum number of samples. If float, it is a fraction of the number of samples.

    min_samples_leaf : int or float, optional, default=1
        The minimum number of samples required to be at a leaf node. If int, it is the minimum number of samples. If float, it is a fraction of the number of samples.

    max_features : int, float or {'sqrt', 'log2'}, optional, default=None
        The number of features to consider when looking for the best split. If None, all features are considered. 

    random_state : int, optional, default=None
        Seed used by the random number generator.
        If None, non-deterministic behaviour will be used.

    max_leaf_nodes : int, optional, default=None
        The maximum number of leaf nodes in the tree.

    min_info_gain : float, optional, default=0.0
        The minimum information gain required to make a split.
    """
    
    def __init__(
        self,
        criterion: str = "gini",
        splitter: Optional[Literal["best", "random"]] = "best",
        max_depth: Optional[int] = None,
        min_samples_split: Optional[Union[int, float]] = 2,
        min_samples_leaf: Optional[Union[int, float]] = 1,
        max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_info_gain: Optional[float] = 0.0):
        
        """
        Initialize the Decision Tree base class with the given hyperparameters.
        """
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
        self.max_leaf_nodes = max_leaf_nodes
        self.min_info_gain = min_info_gain

        self.depth = 1
        self.parent = None
        self.levels = []
        self.nodes = []
        self.n_nodes = 0


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the decision tree to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix for training.

        y : np.ndarray
            The target vector for training.

        This method builds the tree by recursively splitting nodes based on the best feature and threshold.
        """
        self.parent = Node(
            X,
            y,
            self.criterion,
            self.splitter,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_features,
            self.min_info_gain,
        )

        level = Level(self.parent, max_leaf_nodes=self.max_leaf_nodes, leaf_nodes=0)
        self.levels.append(level)

        next_level = level.split()
        while (next_level and (self.max_depth is None or self.depth < self.max_depth)):
            self.depth += 1
            self.levels.append(next_level)
            next_level = next_level.split()

        self.nodes = [item for level in self.levels for item in level.nodes]
        self.n_nodes = len(self.nodes)




    def build_graph(self, node, graph, parent_id = None, node_id = 0, feature_names = None):
        """
        Recursively build a graph structure from the decision tree nodes for visualization.

        Parameters:
        -----------
        node : Node
            The current node in the decision tree.

        graph : networkx.DiGraph
            The directed graph used to store tree structure.

        parent_id : int, optional
            The ID of the parent node, if available.

        node_id : int, optional, default=0
            The unique ID for the current node.

        feature_names: list / np.ndarray, optional, default=None
            Feature names.

        Returns:
        --------
        int
            The updated node ID after processing the current node.
        """
        if node is None:
            return node_id

        current_node_id = node_id
        node_label = (
            f"Samples: {node.n_samples}\n"
            f"Impurity: {node.node_impurity:.2f}\n"
            f"Feature: {feature_names[node.best_split_feature] if (feature_names and node.best_split_feature is not None) else node.best_split_feature}\n"
            f"Threshold: {node.best_split_threshold if node.best_split_threshold is None else (node.best_split_threshold if isinstance(node.best_split_threshold, str) else f'{node.best_split_threshold:.2f}')}"
        )
        graph.add_node(current_node_id, label=node_label)

        if parent_id is not None:
            graph.add_edge(parent_id, current_node_id)

        node_id += 1
        if node.best_split:
            for child in node.best_split:
                node_id = self.build_graph(child, graph, current_node_id, node_id, feature_names)

        return node_id


    def plot_tree(self, feature_names=None):
        """
        Visualize the decision tree using NetworkX and Matplotlib.

        This method uses NetworkX to create a directed graph representation of the decision tree 
        and then uses Matplotlib to plot it.
        
        Parameters:
        -----------
        feature_names: list / np.ndarray, optional, default=None
            Feature names.
        -----------
        Returns:
        None
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            graph = nx.DiGraph()
            self.build_graph(self.parent, graph, feature_names=feature_names)
            pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

            plt.figure(figsize=(12, 8))
            nx.draw(
                graph,
                pos,
                with_labels=True,
                labels=nx.get_node_attributes(graph, "label"),
                node_size=3000,
                node_color="lightblue",
                font_size=5,
                font_weight="bold",
            )
            plt.show()
