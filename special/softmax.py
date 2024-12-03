import numpy as np

def softmax(x):
    """
    Compute the softmax of a list of numbers.

    The softmax function is a mathematical function that converts a vector of numbers
    into a vector of probabilities, where the probabilities of each value are proportional
    to the relative scale of each value in the vector.

    Parameters:
    x (array-like): A list or array of numbers.

    Returns:
    numpy.ndarray: An array of the same shape as `x` containing the softmax probabilities.
    """

    maxx = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - maxx)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)