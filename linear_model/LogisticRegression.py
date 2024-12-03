import numpy as np
from mlkit.metrics import classification_score, log_loss
from mlkit.special import sigmoid
from mlkit.preprocessing import MinMaxScaler

class LogisticRegression:

    """
    A custom implementation of Logistic Regression with support for different 
    optimization methods and regularization techniques.

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
                 n_iters = 25000, 
                 lr = 1e-3, 
                 reg = ("l2", 1e-3), 
                 method = "gd",
                 batch_size = 128,
                 scaler = MinMaxScaler(), 
                 threshold = 0.5,
                 verbose = 1000,
                 random_state = 42):
        
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
        self.threshold = threshold
        self.verbose = verbose

        self.n_samples = None
        self.n_feats = None
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
        self.w = np.zeros(self.n_feats)
        self.b = 0

    def print(self, x):
        """
        Verbose management logic
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
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        """
        y_pred = sigmoid(np.dot(X, self.w) + self.b)
        self.w_gradient = np.dot((y_pred - y), X) / self.n_samples
        self.b_gradient = np.mean((y_pred  - y))
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
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
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
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        """
        for i in range(self.n_iters + 1):
            if self.verbose != 0 and i % self.verbose == 0:
                y_pred = sigmoid(np.dot(X, self.w) + self.b)
                loss = log_loss(y, y_pred)
                self.print(f"Iter: {i}. Log Loss: {loss}.")
            self._gradient_step(X, y)

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
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
        self._init_params()
        self.print("Starting training the model:\n")
        self._gradient_descent(X, y)
        self.print("\nDone.")
        self.is_fitted = True
        self.print(f"\nBest weights: {self.w}.")
        self.print(f"Best bias: {self.b}.")
        self.print("\nEval metrics on train data:")
        y_pred = (sigmoid(np.dot(X, self.w) + self.b) > self.threshold).astype(int)
        self.print(classification_score(y, y_pred))


    def predict(self, X):
        """
        Predicts the binary class labels for the given input data.

        This method computes the predicted labels (0 or 1) based on the learned weights (w) and bias (b).
        If a scaler is provided, the input data is first scaled using the scaler.

        Args:
            X (array-like): Input data to make predictions on, with shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted binary labels (0 or 1) for each sample in the input data.
        
        Raises:
            AssertionError: If the model has not been fitted (self.is_fitted is False).
        """
        assert self.is_fitted, "Please fit the model first."
        if self.scaler:
            X = self.scaler.transform(X)
        return (sigmoid(np.dot(X, self.w) + self.b) > self.threshold).astype(int)


    def predict_proba(self, X):
        """
        Predicts the probability of the positive class (1) for the given input data.

        This method computes the probabilities for both class 0 and class 1, with the positive class probability 
        being calculated using the sigmoid function on the linear combination of input features and learned parameters.
        If a scaler is provided, the input data is first scaled using the scaler.

        Args:
            X (array-like): Input data to calculate probabilities for, with shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Probability estimates for both classes (0 and 1) for each sample, 
                            with shape (n_samples, 2).
        
        Raises:
            AssertionError: If the model has not been fitted (self.is_fitted is False).
        """
        assert self.is_fitted, "Please fit the model first."
        if self.scaler:
            X = self.scaler.transform(X)
        p1 = sigmoid(np.dot(X, self.w) + self.b)
        p0 = 1 - p1
        return np.hstack([p0[:, np.newaxis], p1[:, np.newaxis]])


        

    def score(self, X, y):
        """
        Compute regression performance metrics.

        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            True target values

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
        return classification_score(y, y_pred)

    def get_params(self):
        return self.w, self.b

    def get_gradients(self):
        return self.w_gradient, self.b_gradient
    

    def reset(self):
        self._init_params()
        self.is_fitted = False