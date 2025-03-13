import numpy as np


def step(x: float):
    """
    Step function (binary step or Heaviside step function).

    Parameters:
    - x (float): Input value.

    Returns:
    - int: 1 if x > 0, else 0.
    """
    return 1 if x > 0 else 0


def sigmoid(x: float):
    """
    Sigmoid function.

    Parameters:
    - x (float): Input value.

    Returns:
    - float: Result of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def tanh(x: float):
    """
    Hyperbolic tangent function.

    Parameters:
    - x (float): Input value.

    Returns:
    - float: Result of the hyperbolic tangent function.
    """
    return np.tanh(x)


def relu(x: float):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
    - x (float): Input value.

    Returns:
    - float: Result of ReLU activation.
    """
    return max(0, x)


def leaky_relu(x: float, alpha=0.01):
    """
    Leaky ReLU activation function.

    Parameters:
    - x (float): Input value.
    - alpha (float, optional): Slope for negative values. Default is 0.01.

    Returns:
    - float: Result of Leaky ReLU activation.
    """
    return max(alpha * x, x)


def prelu(x: float, alpha=0.01):
    """
    Parametric ReLU (PReLU) activation function.

    Parameters:
    - x (float): Input value.
    - alpha (float, optional): Learnable parameter for negative values. Default is 0.01.

    Returns:
    - float: Result of PReLU activation.
    """
    return x if x > 0 else alpha * x


def elu(x: float, alpha=1.0):
    """
    Exponential Linear Unit (ELU) activation function.

    Parameters:
    - x (float): Input value.
    - alpha (float, optional): Slope for negative values. Default is 1.0.

    Returns:
    - float: Result of ELU activation.
    """
    return x if x > 0 else alpha * (np.exp(x) - 1)


def softmax(x: float):
    """
    Softmax activation function for an array.

    Parameters:
    - x (np.ndarray): Input array.

    Returns:
    - np.ndarray: Result of the softmax activation.
    """
    expo = np.exp(x)
    expo_sum = np.sum(np.exp(x))
    return expo/expo_sum


def linear(x: float):
    """
    Linear activation function (identity function).

    Parameters:
    - x (float): Input value.

    Returns:
    - float: Same as the input value.
    """
    return x


def swish(x: float, beta=1.0):
    """
    Swish activation function.

    Parameters:
    - x (float): Input value.
    - beta (float, optional): Swish beta parameter. Default is 1.0.

    Returns:
    - float: Result of the swish activation.
    """
    return x / (1 + np.exp(-beta * x))
