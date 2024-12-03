import numpy as np
from mlkit.metrics import regression_score
from mlkit.preprocessing import MinMaxScaler

class LinearRegression:
    """
    A custom implementation of Linear Regression with support for different 
    optimization methods and regularization techniques. 
    Has support for multi-label Regression.

    Parameters:
    -----------
    n_iters : int, default = 25000
        Number of iterations for gradient descent.
    lr : float, default = 1e-3
        Learning rate for gradient descent.
    reg : tuple or None, default = ("l2", 1e-3)
        Regularization specification. 
        - Format: ("l1", lambda) for Lasso
        - Format: ("l2", lambda) for Ridge
        - Format: ("elasticnet", lambda, alpha) for combined L1/L2 penalty
    method : str, default = 'gd'
        Optimization method. Either 'gd' (Gradient Descent) or 'sgd' (Stochastic Batch Gradient Descent).
    batch_size : int, default = 128
        Mini batch size in one gradient step. Used only if method is 'sgd'.
    scaler : object, default = MinMaxScaler()
        Scaler transformer to use.
    verbose : int, default = 1000
        Frequency of printing training progress (0 means no output).
    random_state : int, default = 42
        Random state seed used by Numpy.
    """
    
    def __init__(self, 
                 n_iters=25000, 
                 lr=1e-3, 
                 reg=("l2", 1e-3), 
                 method="gd",
                 batch_size=128,
                 scaler=MinMaxScaler(), 
                 verbose=1000,
                 random_state=42):
        
        self.n_iters = n_iters
        self.lr = lr
        self.reg = reg
        self.method = method
        if method == "sgd":
            self.n_iters = 100000
            self.lr = 1e-2 
        self.batch_size = batch_size
        if scaler:
            self.scaler = type(scaler)()
        else:
            self.scaler = None
        self.verbose = verbose

        self.n_samples = None
        self.n_feats = None
        self.n_labels = None
        self.w = None
        self.w_gradient = None
        self.b = None
        self.b_gradient  = None

        self.is_fitted = False
        np.random.seed(random_state)


    def _init_params(self):
        """
        Initialize model parameters with zero values.
        """
        self.w = np.zeros((self.n_labels, self.n_feats))
        self.b = np.zeros(self.n_labels)

    def print(self, x):
        """
        Verbose management logic.
        """
        if self.verbose:
            print(x)
        else:
            pass

    def _compute_gradient(self, X, y):
        """
        Compute gradients for weights and bias.
        
        Includes optional L1, L2 or ElasticNet regularization.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_feats)
            Input features
        y : numpy.ndarray of shape (n_samples, n_labels)
            Target values
        """
        y_pred = np.dot(X, self.w.T) + self.b
        self.w_gradient = 2 * np.dot((y_pred - y).T, X) / self.n_samples # (n_labels, n_samples) * (n_samples, n_feats) => (n_labels, n_feats)
        self.b_gradient = np.mean(2 * (y_pred  - y), axis = 0)
        if self.reg is not None:
            if self.reg[0] == "l1":
                self.w_gradient += self.reg[1] / self.n_samples * np.sign(self.w)
            elif self.reg[0] == "l2":
                self.w_gradient += self.reg[1] / self.n_samples * (2 * self.w)
            elif self.reg[0] == "elasticnet":
                self.w_gradient += self.reg[1] / self.n_samples * (self.reg[2] * np.sign(self.w) + (1 - self.reg[2]) * (2 * self.w))
                

    def _gradient_step(self, X, y):
        """
        Perform a single gradient descent step.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_feats)
            Input features
        y : numpy.ndarray of shape (n_samples, n_labels)
            Target values
        """
        if self.method == "gd":
            self._compute_gradient(X, y)
        elif self.method == "sgd":
            # Shuffle the indices for stochastic gradient descent (SGD)
            indices = np.random.permutation(len(X))
            batch_indices = indices[:self.batch_size]  # Get a batch from the shuffled indices
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            self._compute_gradient(X_batch, y_batch)
        self.w -= self.lr * self.w_gradient
        self.b -= self.lr * self.b_gradient

    def _gradient_descent(self, X, y):
        """
        Perform gradient descent optimization.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_feats)
            Input features
        y : numpy.ndarray of shape (n_samples, n_labels)
            Target values
        """
        for i in range(self.n_iters + 1):
            if self.verbose != 0 and i % self.verbose == 0:
                y_pred = np.dot(X, self.w.T) + self.b # (n_samples, n_feats) * (n_feats, n_labels)
                rse = (y_pred - y) ** 2
                self.print(f"Iter: {i}. RSS: {np.sum(rse)}. MSE: {np.mean(rse)}.")
            self._gradient_step(X, y)


    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_feats)
            Input features
        y : numpy.ndarray of shape (n_samples, n_labels)
            Target values

        Returns:
        --------
        self : LinearRegression
            Fitted model
        """
        if self.scaler:
            self.scaler = type(self.scaler)()
            X = self.scaler.fit_transform(X)
        self.n_samples, self.n_feats = len(X), len(X[0])
        if y.ndim == 1:
            y = np.expand_dims(y, axis=-1)
        self.n_labels = len(y[0])
        self._init_params()
        self.print("Starting training the model:\n")
        self._gradient_descent(X, y)
        self.print("\nDone.")
        self.print(f"\nBest weights: {self.w}.")
        self.print(f"Best bias: {self.b}.")
        self.print("\nEval metrics on train data")
        y_pred = np.dot(X, self.w.T) + self.b # (n_samples, n_feats) * (n_feats, n_labels) => (n_samples, n_labels)
        self.print(regression_score(y, y_pred))
        self.is_fitted = True



    def predict(self, X):
        assert self.is_fitted, "Please fit the model first."
        if self.scaler:
            X = self.scaler.transform(X)
        return np.dot(X, self.w.T) + self.b # (n_samples, n_feats) * (n_feats, n_labels)



    def score(self, X, y):
        """
        Compute regression performance metrics.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_feats)
            Input features
        y : numpy.ndarray of shape (n_samples, n_labels)
            Target values

        Returns:
        --------
        dict
            Regression performance metrics

        Raises:
        -------
        AssertionError
            If the model has not been fitted
        """
        assert self.is_fitted, "Please fit the model first."
        y_pred = self.predict(X)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=-1)
        return regression_score(y, y_pred)


    def get_params(self):
        return self.w, self.b

    def get_gradients(self):
        return self.w_gradient, self.b_gradient
    
    def reset(self):
        self._init_params()
        self.is_fitted = False