import numpy as np
import pandas as pd

def train_test_split(*args, train_size=None, test_size=None, random_state=42):

    """
    Split multiple input sequences into training and testing sets.

    Parameters:
    -----------
    *args : sequence of arrays or lists
        Sequences to be split. All must have the same length.
    
    train_size : float, optional
        Proportion of data for the training set (between 0 and 1).
    
    test_size : float, optional
        Proportion of data for the test set (between 0 and 1).
    
    random_state : int, optional
        Seed for random shuffling.

    Returns:
    --------
    tuple of tuples
        Tuple containing the training and testing sets for each input sequence.

    Raises:
    -------
    AssertionError
        If neither `train_size` nor `test_size` is provided, or if input sequences have different lengths.
    """

    assert train_size or test_size, "Please provide either train_size or test_size."
    assert len(np.unique([len(arg) for arg in args])) == 1, "Input sequences must have the same length."
    
    if train_size:
        test_size = 1 - train_size
    if test_size:
        train_size = 1 - test_size
    
    np.random.seed(random_state)
    
    n_samples = len(args[0])
    permuted_indices = np.random.permutation(n_samples)
    
    train_indices, test_indices = permuted_indices[:int(n_samples * train_size)], permuted_indices[int(n_samples * train_size):]
    
    splitted = []
    for arg in args:
        assert isinstance(arg, (np.ndarray, list, pd.DataFrame)), "Input sequence must be: np.ndarray, list, or pd.DataFrame"
        if isinstance(arg, pd.DataFrame):
            splitted.append(arg.iloc[train_indices])
            splitted.append(arg.iloc[test_indices])
        else:
            splitted.append(arg[train_indices])
            splitted.append(arg[test_indices])
        
    return tuple(splitted)
