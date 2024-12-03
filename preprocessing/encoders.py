import numpy as np
import pandas as pd

class LabelEncoder:
    """
    A class used to encode labels into one-hot encoding and decode them back to original labels.
    Attributes
    ----------
    initial_labels : list
        A list of unique labels extracted from the input data.
    labels : list
        A list of integer labels corresponding to the unique labels.
    n_labels : int
        The number of unique labels.
    mapping : dict
        A dictionary mapping each unique label to an integer.

        
    Methods
    -------
    fit(y)
        Fits the encoder to the given labels.
    transform(y)
        Transforms labels into one-hot encoding.
    fit_transform(y)
        Fits the encoder and transforms the data in one step.
    inverse_transform(y_enc)
        Converts one-hot encoded labels back to original labels.

    """

    def __init__(self):
        self.initial_labels = None
        self.labels = None
        self.n_labels = None
        self.mapping = None
        self.mapping_inv = None

    def fit(self, y):
        """Fit the encoder to the given labels."""
        # Extract unique labels from y
        self.initial_labels = list(pd.unique(y))
        # Create a list of integer labels starting from 0
        self.labels = list(range(len(self.initial_labels)))
        self.n_labels = len(self.labels)
        # Map each label to a unique integer
        self.mapping = dict(zip(self.initial_labels, self.labels))
        self.mapping_inv = dict(zip(self.labels, self.initial_labels))

    def transform(self, y):
        """Transform labels into one-hot encoding."""
        # Create an empty array to hold the one-hot encoded data
        y_enc = np.zeros((len(y), self.n_labels), dtype=int)
        for i in range(len(y)):
            # Find the integer index for the label y[i]
            one_pos = self.mapping[y[i]]
            # Set the corresponding position in the one-hot array to 1
            y_enc[i, one_pos] = 1
        return y_enc

    def fit_transform(self, y):
        """Fit the encoder and transform the data in one step."""
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_enc):
        """Convert one-hot encoded labels back to original labels."""
        # Find the index where 1 is located for each row and map it back to the original label
        return [self.initial_labels[np.argmax(row)] for row in y_enc]
    
    def inverse_mapping(self, y_enc_discrete):
        """Convert discrete encoded labels back to original labels."""
        return np.vectorize(self.mapping_inv.get)(y_enc_discrete)