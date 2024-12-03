import numpy as np
def sigmoid(x):
    """
    Compute the sigmoid of x.

    The sigmoid function is defined as 1 / (1 + exp(-x)), where exp is the exponential function.

    Parameters:
    x (float or np.ndarray): The input value or array of values.

    Returns:
    float or np.ndarray: The sigmoid of the input value(s).
    """
    return 1 / (1 + np.exp(-x))