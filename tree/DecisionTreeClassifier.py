import numpy as np
from scipy.ndimage import shift
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
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
        while node.best_split and node.best_split_info_gain > 0:
            if not node.best_split:  # Continue traversing while the node has a valid split
                break
            if type(node.best_split_threshold) != str:
                if sample[node.best_split_feature] > node.best_split_threshold:
                    node = node.best_split[0]
                else:
                    node = node.best_split[1]
            else:
                if sample[node.best_split_feature] == node.best_split_threshold:
                    node = node.best_split[0]
                else:
                    node = node.best_split[1]

        # For leaf nodes, return the most common class
        # Add a safety check to prevent IndexError
        if len(node.y) > 0:
            return Counter(node.y).most_common(1)[0][0]
        else:
            # If no samples in the node, return a default value (e.g., most common class in the root)
            return Counter(self.parent.y).most_common(1)[0][0]
        
        
    def predict(self, X):
        return np.array([self._traverse_tree(sample, self.parent) for sample in X])
    
    def score(self, X, y):
        preds = self.predict(X)
        return classification_score(y, preds)
    
    # def plot_tree(self, figsize=(20, 10), node_size=3000, font_size=10):
    #     """
    #     Visualize the decision tree structure.
        
    #     Parameters:
    #     -----------
    #     figsize : tuple, optional (default=(20, 10))
    #         Size of the figure to plot
    #     node_size : int, optional (default=3000)
    #         Size of the nodes in the plot
    #     font_size : int, optional (default=10)
    #         Font size for node labels
    #     """
    #     # Create a directed graph
    #     G = nx.DiGraph()
        
    #     def add_nodes_and_edges(node, parent_id=None, edge_label=None):
    #         # Generate a unique identifier for the current node
    #         node_id = id(node)
            
    #         # Create node label
    #         if node.best_split_feature is not None:
    #             # For non-leaf nodes, show the splitting feature and threshold
    #             if type(node.best_split_threshold) != str:
    #                 node_label = (f"Feature {node.best_split_feature}\n"
    #                             f"Threshold > {node.best_split_threshold:.2f}\n"
    #                             f"Samples: {node.n_samples}")
    #             else:
    #                 node_label = (f"Feature {node.best_split_feature}\n"
    #                             f"Threshold = {node.best_split_threshold}\n"
    #                             f"Samples: {node.n_samples}")
    #         else:
    #             # For leaf nodes, show the most common class
    #             node_label = (f"Leaf\n"
    #                         f"Class: {Counter(node.y).most_common(1)[0][0]}\n"
    #                         f"Samples: {node.n_samples}")
            
    #         # Add node to the graph
    #         G.add_node(node_id, label=node_label)
            
    #         # Add edge from parent if exists
    #         if parent_id is not None:
    #             G.add_edge(parent_id, node_id, label=edge_label)
            
    #         # Recursively add child nodes
    #         if node.best_split:
    #             add_nodes_and_edges(node.best_split[0], node_id, "True")
    #             add_nodes_and_edges(node.best_split[1], node_id, "False")
        
    #     # Start adding nodes from the root
    #     add_nodes_and_edges(self.parent)
        
    #     # Create the plot
    #     plt.figure(figsize=figsize)
    #     pos = nx.spring_layout(G, k=0.9, iterations=50)
        
    #     # Draw nodes
    #     nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.9)
        
    #     # Draw edges
    #     nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        
    #     # Draw node labels
    #     node_labels = nx.get_node_attributes(G, 'label')
    #     nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size, font_weight="bold")
        
    #     # Draw edge labels
    #     edge_labels = nx.get_edge_attributes(G, 'label')
    #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
    #     plt.title("Decision Tree Visualization")
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.show()

    def build_graph(self, node, graph, parent_id=None, node_id=0):
        """
        Recursively build a graph from the decision tree nodes.

        Parameters:
        - node: The current tree node
        - graph: The NetworkX graph object to add nodes and edges
        - parent_id: The ID of the parent node
        - node_id: The unique ID for the current node

        Returns:
        - The next available unique node ID
        """
        if node is None:
            return node_id

        # Create a unique identifier for the current node
        current_node_id = node_id

        # Create a label for the node with its properties
        node_label = (
            f"Samples: {node.n_samples}\n"
            f"Impurity: {node.node_impurity:.2f}\n"
            f"Feature: {node.best_split_feature}\n"
            f"Threshold: {node.best_split_threshold if node.best_split_threshold is None else (node.best_split_threshold if isinstance(node.best_split_threshold, str) else f'{node.best_split_threshold:.2f}')}"
        )

        # Add the current node to the graph
        graph.add_node(current_node_id, label=node_label)

        # Connect the node to its parent
        if parent_id is not None:
            graph.add_edge(parent_id, current_node_id)

        # Increment the node ID for the next node
        node_id += 1

        # Recursively process child nodes
        if node.best_split:
            for child in node.best_split:
                node_id = self.build_graph(child, graph, current_node_id, node_id)

        return node_id

    def plot_tree(self):
        """
        Plot the decision tree using NetworkX and Matplotlib.
        """
        # Create a directed graph
        graph = nx.DiGraph()

        # Build the graph starting from the root node
        self.build_graph(self.parent, graph)

        # Generate positions for nodes using Graphviz layout
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

        # Plot the graph
        plt.figure(figsize=(12, 8))
        nx.draw(
            graph,
            pos,
            with_labels=True,
            labels=nx.get_node_attributes(graph, 'label'),
            node_size=2000,
            node_color="lightblue",
            font_size=8,
            font_weight="bold"
        )
        plt.show()






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
        self.best_split_impurity = float('inf')
        self.best_split_info_gain = 0
    


    def compute_gini_impurity(self):
        imp = 1
        for value in np.unique(self.y):
            occurence = np.sum((self.y == value))
            imp -= (occurence / len(self.y)) ** 2
        self.node_impurity = imp


    def split_by(self, feature, threshold): # binary tree
        if type(threshold) != str:
            mask = (self.X[:, feature]) > threshold
        else:
            mask = (self.X[:, feature]) == threshold
        X_above, y_above = self.X[mask], self.y[mask]
        X_below, y_below = self.X[~mask], self.y[~mask]
        node_1 = Node(X_above, y_above) # > threshold (for nums) / == threshold (for cat)
        node_2 = Node(X_below, y_below) # <= threshold (for nums) / != threshold (for cat)
        return node_1, node_2
    

    def split(self):
        for feature in range(len(self.X[0])):
            if type(self.X[0, feature]) != str:
                possible_thresholds = ((shift(np.unique(self.X[:, feature]), 1) + np.unique(self.X[:, feature])) / 2)[1:]
            else:
                possible_thresholds = np.unique(self.X[:, feature])
            for threshold in possible_thresholds:
                node_1, node_2 = self.split_by(feature, threshold)
                # Ensure we don't create empty nodes
                if len(node_1.y) == 0 or len(node_2.y) == 0:
                    continue
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