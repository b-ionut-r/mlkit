class Level:
    """
    A class used to represent a Level in a decision tree.
    Attributes
    ----------
    nodes : list
        A list of nodes at the current level of the decision tree.
    max_leaf_nodes: int or None
        The maximum number of leaf nodes allowed. If None, there is no limit.
    leaf_nodes: int
        Leaf nodes counter.

    Methods:
    -------
    __init__(*nodes, max_leaf_nodes)
        Initializes the Level with the given nodes and max_leaf_nodes.
    split()
        Splits the nodes at the current level and returns a new Level with the resulting nodes if any splits are possible.
    """

    def __init__(self, *nodes, max_leaf_nodes, leaf_nodes=0):
        self.nodes = list(nodes)
        self.max_leaf_nodes = max_leaf_nodes  # Store max_leaf_nodes in the class
        self.leaf_nodes = leaf_nodes  # Initialize leaf nodes counter, defaulting to 0

    def split(self):
        new_nodes = []
        for node in self.nodes:
            # Try to split the node
            splits = node.split()
            # If splits are possible, add them
            if splits:
                new_nodes.extend(splits)
            # If not, increment leaf nodes counter
            else:
                self.leaf_nodes += 1
            # Stop node splitting if leaf nodes exceed maximum allowed, unless max_leaf_nodes is None
            if self.max_leaf_nodes and self.leaf_nodes >= self.max_leaf_nodes:
                break

        # Only return a new level if splits were found
        return Level(*new_nodes, max_leaf_nodes=self.max_leaf_nodes, leaf_nodes=self.leaf_nodes) if new_nodes else None
