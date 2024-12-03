import numpy as np
from mlkit.metrics import classification_score, multi_log_loss
from mlkit.special import softmax
from mlkit.preprocessing import MinMaxScaler
from mlkit.preprocessing import LabelEncoder

class MultiLogisticRegression:

    """
    A custom implementation of Multiclass Logistic Regression with support for different 
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
        Optimization method. Either 'gd' (Gradient Descent) or 'sgd' (Stochastic (Batch) Gradient Descent).
    batch_size : int, default = 128
        Mini batch size in one gradient step. Used only if method is 'sgd'.
    scaler : object, default = MinMaxScaler()
        Scaler transformer to use.
    label_encoder : object, default = LabelEncoder()
        Label encoder transformer to use.
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
                 label_encoder = LabelEncoder(),
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
        self.label_encoder = label_encoder
        self.verbose = verbose

        self.n_samples = None
        self.n_feats = None
        self.n_classes = None
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
        # self.w = np.zeros((self.n_classes, self.n_feats))
        self.w = np.random.randn(self.n_classes, self.n_feats) * 0.01
        self.b = np.zeros(self.n_classes)

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
        X : numpy.ndarray of shape (n_samples, n_feats)
            Input features
        y : numpy.ndarray of shape (n_samples, n_classes)
            Target values
        """
        y_pred = softmax(np.dot(X, self.w.T) + self.b)
        self.w_gradient = np.dot((y_pred - y).T, X) / self.n_samples # (n_classes, n_samples) * (n_samples, n_feats) => (n_classes, n_feats)
        self.b_gradient = np.mean((y_pred - y), axis = 0)
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
        y : numpy.ndarray of shape (n_samples, n_classes)
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
        X : numpy.ndarray of (n_samples, n_feats)
            Input features
        y : numpy.ndarray of (n_samples, n_classes)
            Target values
        """
        for i in range(self.n_iters + 1):
            if self.verbose != 0 and i % self.verbose == 0:
                y_pred = softmax(np.dot(X, self.w.T) + self.b)
                loss = multi_log_loss(y, y_pred)
                self.print(f"Iter: {i}. Log Loss: {loss}.")
            self._gradient_step(X, y)


    def fit(self, X, y):
        """
        Fit the multiclass logistic regression model to the training data.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_feats)
            Input features
        y : numpy.ndarray of shape (n_samples, n_classes)
            Target values
        """
    
        if self.scaler:
            self.scaler = type(self.scaler)() 
            X = self.scaler.fit_transform(X)
        self.n_samples, self.n_feats = len(X), len(X[0])

        y_enc = self.label_encoder.fit_transform(y)
        self.n_classes = self.label_encoder.n_labels


        self._init_params()
        self.print("Starting training the model:\n")
        self._gradient_descent(X, y_enc)
        self.print("\nDone.")
        self.is_fitted = True
        self.print(f"\nBest weights: {self.w}.")
        self.print(f"Best bias: {self.b}.")
        self.print("\nEval metrics on train data:")
        y_pred = np.squeeze(np.argmax((softmax(np.dot(X, self.w.T) + self.b)), axis=-1))
        y_pred = self.label_encoder.inverse_mapping(y_pred)
        self.print(classification_score(y, y_pred))


    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Parameters:
        X (numpy.ndarray): The input data to predict, where each row represents a sample and each column represents a feature.

        Returns:
        numpy.ndarray: The predicted class labels for each sample in the input data.

        Raises:
        AssertionError: If the model is not fitted before calling this method.
        """
        assert self.is_fitted, "Please fit the model first."
        if self.scaler:
            X = self.scaler.transform(X)
        y_pred = np.squeeze(np.argmax((softmax(np.dot(X, self.w.T) + self.b)), axis=-1))
        y_pred = self.label_encoder.inverse_mapping(y_pred)
        return y_pred



    def predict_proba(self, X):
        """
        Predict probability estimates for the given input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data for which to predict probability estimates.

        Returns:
        --------
        array-like of shape (n_samples, n_classes)
            The predicted probability estimates for each class.

        Raises:
        -------
        AssertionError
            If the model is not fitted before calling this method.
        """
        assert self.is_fitted, "Please fit the model first."
        if self.scaler:
            X = self.scaler.transform(X)
        return softmax(np.dot(X, self.w.T) + self.b)


        

    def score(self, X, y):
        """
        Computes multi-class classification performance metrics
        (using MACRO average).

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_feats)
            Input features
        y : numpy.ndarray of shape (n_samples, n_classes)
            True target values

        Returns:
        --------
        dict
            Multiclass Classification performance metrics.

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