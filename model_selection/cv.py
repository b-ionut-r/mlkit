import numpy as np

class KFold:
    """
    K-Folds cross-validator.
    Provides train/test indices to split data into train/test sets. Split dataset into k consecutive folds (without shuffling by default).
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting into batches.
    random_state : int, default=42
        When shuffle is True, random_state affects the ordering of the indices, which controls the randomness of each fold.
    Methods
    -------
    split(X)
        Generate indices to split data into training and test set.
    Examples
    --------
    >>> import numpy as np
    >>> from mlkit.model_selection.cv import KFold
    >>> X = np.arange(10)
    >>> kf = KFold(n_splits=5)
    >>> for train_index, test_index in kf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [2 3 4 5 6 7 8 9] TEST: [0 1]
    TRAIN: [0 1 4 5 6 7 8 9] TEST: [2 3]
    TRAIN: [0 1 2 3 6 7 8 9] TEST: [4 5]
    TRAIN: [0 1 2 3 4 5 8 9] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
    """


    def __init__(self, n_splits = 5, shuffle = True, random_state = 42):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        np.random.seed(self.random_state)
        idx_perm = np.random.permutation(len(X)) if self.shuffle else np.arange(len(X))
        val_size = len(X) // self.n_splits
        for i in range(self.n_splits):
            start = i * val_size
            finish = (i + 1) * val_size if i != self.n_splits - 1 else len(X)
            val_idx = idx_perm[start:finish]
            train_idx = np.concatenate([idx_perm[:start], idx_perm[finish:]])
            yield train_idx, val_idx


