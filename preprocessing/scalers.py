import pandas as pd
import numpy as np


class MinMaxScaler:

    """
    MinMaxScaler scales the input features to a specified range.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    mins: list
        List of minimum values of the features.

    maxs: list
        List of maximum values of the features.

    Methods:
    --------
    fit(X):
        Fit the scaler on the input data.

    transform(X):
        Transform the input data.

    fit_transform(X):
        Fit and transform the input data.

    """


    def __init__(self):

        self.mins = []
        self.maxs = []


    def fit(self, X):

        if isinstance(X, (pd.DataFrame, list)):
            try:
                X = np.array(X)
            except:
                raise Exception("Input can't be converted to np.ndarray. Please check if it is of accepted dtype: pd.DataFrame or list.")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        for column in X.T:
            min_val = column.min()
            max_val = column.max()
            self.mins.append(min_val)
            self.maxs.append(max_val)


    def transform(self, X):

        assert len(self.mins), "Please fit the scaler first."

        if isinstance(X, (pd.DataFrame, list)):
            try:
                X = np.array(X)
            except:
                raise Exception("Input can't be converted to np.ndarray. Please check if it is of accepted dtype: pd.DataFrame or list.")
        
        assert len(X[0]) == len(self.mins), "Number of features mismatch."

        X_transformed = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            range_val = self.maxs[i] - self.mins[i]
            if range_val == 0:
                X_transformed[:, i] = 0  # Handle constant column (no scaling)
            else:
                X_transformed[:, i] = (X[:, i] - self.mins[i]) / range_val

        return X_transformed
 
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    






class StandardScaler:

    """

    A class to standardize the features of a dataset (scale to zero mean and unit variance).

    Attributes:
    means (list): List of mean values for each feature.
    stddevs (list): List of standard deviation values for each feature.

    Methods:
    fit(X): Computes the mean and standard deviation for each feature in the dataset.
    transform(X): Standardizes the input data using the computed means and standard deviations.
    fit_transform(X): Fits the scaler to the data and then transforms the data.
    
    """


    def __init__(self):

        self.means = []
        self.stddevs = []


    def fit(self, X):

        if isinstance(X, (pd.DataFrame, list)):
            try:
                X = np.array(X)
            except:
                raise Exception("Input can't be converted to np.ndarray. Please check if it is of accepted dtype: pd.DataFrame or list.")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        for column in X.T:
            mean_val = column.mean()
            std_val = column.std()
            self.means.append(mean_val)
            self.stddevs.append(std_val)


    def transform(self, X):

        assert len(self.means), "Please fit the scaler first."

        if isinstance(X, (pd.DataFrame, list)):
            try:
                X = np.array(X)
            except:
                raise Exception("Input can't be converted to np.ndarray. Please check if it is of accepted dtype: pd.DataFrame or list.")
        
        assert len(X[0]) == len(self.means), "Number of features mismatch."

        X_transformed = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            mean, std = self.means[i], self.stddevs[i]
            if std == 0:
                X_transformed[:, i] = 0 # Avoid division by zero
            else:
                X_transformed[:, i] = (X[:, i] - mean) / std

        return X_transformed
 
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)





        
                
        
