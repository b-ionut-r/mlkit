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
    



class OneHotEncoder:
   
    def __init__(self, cat_idx="auto"):
        """
        Initialize the OneHotEncoder.
        Parameters:
        cat_idx (str or list, optional): Specifies the indices of categorical columns.
                        If "auto", the encoder will automatically detect
                        categorical columns. Default is "auto".
        """
        self.cat_idx = cat_idx
        self.feature_encoders = {}

    def fit(self, X):
        """
        Fit the encoder to the provided data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data to fit the encoder on. Each column represents a feature.

        Notes:
        ------
        - If `self.cat_idx` is a list, it should contain the indices of the categorical columns.
          The method will fit a LabelEncoder to each of these columns.
        - If `self.cat_idx` is set to "auto", the method will automatically detect categorical columns
          by attempting to convert each column to a numeric type. Columns that cannot be converted
          will be treated as categorical and encoded accordingly.
        - The fitted encoders are stored in `self.feature_encoders` dictionary with column indices as keys.
        """
        if isinstance(self.cat_idx, list):  # If categorical column indices are provided
            for i in self.cat_idx:
                encoder = LabelEncoder()
                encoder.fit(X[:, i])  # Fit the LabelEncoder on the column
                self.feature_encoders[i] = encoder
        elif self.cat_idx == "auto":  # Automatically detect categorical columns
            self.cat_idx = []
            i = 0
            while i < X.shape[1]:
                try:
                    # Try converting column to numeric, if it succeeds, treat as numeric
                    X[:, i] = X[:, i].astype(np.float64)
                    i += 1 
                except ValueError:
                    # If ValueError occurs, treat as categorical and encode it
                    self.cat_idx.append(i)
                    encoder = LabelEncoder()
                    encoder.fit(X[:, i])
                    self.feature_encoders[i] = encoder 
                    i += encoder.n_labels  # Skip over newly encoded columns

    def transform(self, X):
        """
        Transforms the input data by applying one-hot encoding to the categorical features.
        Parameters:
        X (numpy.ndarray): The input data to transform.
        Returns:
        numpy.ndarray: The transformed data with one-hot encoded categorical features.
        Raises:
        AssertionError: If the feature_encoders attribute is not set, indicating that the model has not been fitted.
        """
        assert self.feature_encoders, "Please fit the model first."
        X_copy = X.copy()
        offset = 0  # Tracks index shifts due to inserted columns

        for i in self.cat_idx:
            # Transform the column into one-hot encoding
            one_hot_columns = self.feature_encoders[i].transform(X_copy[:, i + offset])
            # Remove the original categorical column
            X_copy = np.hstack(
                (X_copy[:, :i + offset], one_hot_columns, X_copy[:, i + offset + 1:])
            )
            # Adjust offset for the inserted columns
            offset += one_hot_columns.shape[1] - 1

        return X_copy



    def fit_transform(self, X):
        """
        Fits the encoder to the input data and transforms it in a single step.
        
        This method combines the functionality of the `fit` and `transform` methods. 
        It first determines the categorical columns (if `cat_idx` is set to `"auto"`) 
        or uses the provided categorical indices, fits label encoders to these columns, 
        and then applies one-hot encoding to transform the input data.

        Parameters
        ----------
        X : numpy.ndarray
            The input data to be fitted and transformed. 
            It should be a 2D array where rows represent samples and columns represent features.

        Returns
        -------
        numpy.ndarray
            The transformed data, with categorical columns replaced by their 
            one-hot encoded representations.

        Example
        -------
        >>> X = np.array([
        ...     [1, "a", 2],
        ...     [2, "b", 3],
        ...     [0, "a", 2]
        ... ])
        >>> encoder = OneHotEncoder(cat_idx="auto")
        >>> X_encoded = encoder.fit_transform(X)
        >>> print(X_encoded)
        array([[1., 1., 0., 2.],
               [2., 0., 1., 3.],
               [0., 1., 0., 2.]])
        """
        self.fit(X)
        return self.transform(X)



    def inverse_transform(self, X_enc):
        """
        Converts one-hot encoded data back to the original categorical labels.
        
        Parameters:
        X_enc (numpy.ndarray): The one-hot encoded data to be transformed back.
        
        Returns:
        numpy.ndarray: The original categorical labels.
        """
        X_decoded = np.zeros((X_enc.shape[0], X_enc.shape[1] - sum([self.feature_encoders[col].n_labels - 1 for col in self.cat_idx])), dtype=object)
        offset = 0
        for i in range(X_decoded.shape[1]):
            if i in self.cat_idx:
                one_hot_columns = X_enc[:, offset:offset + self.feature_encoders[i].n_labels]
                X_decoded[:, i] = self.feature_encoders[i].inverse_transform(one_hot_columns)
                offset += self.feature_encoders[i].n_labels
            else:
                X_decoded[:, i] = X_enc[:, offset]
                offset += 1
        return X_decoded
