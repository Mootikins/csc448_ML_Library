import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def linear_regression(X, y, n_iter=500, learning_rate=0.01, logging=False, log=None):
    """
    Fit the linear regression for input variable X to output variable y

    :param X: ndarray of floats of shape (num_features, num_data_points)
    :param y: ndarray of floats of shape (num_data_points, 1)
    :param n_iter: int, number of iterations to perform given it does not
        correctly approach the bottom
    :param learning_rate: float, should be between 0 and 1. Determines how
        "aggressive" the weight gradient application is
    :param logging: Boolean, defaults to false to not log iterations, setting to
        true requires a list or other type that has `append` defined that will
        mutate the passed variable
    :param log: a list or other mutable type with `append` defined that will be
        used to backpropogate the progression of the stepping to the user

    :return weights as a numpy array and the bias. If the input data was one
        dimensional, then the weight is the slope and the bias is the y-intercept

    :raises AttributeError: if logging is enabled but no log variable is passed
    """
    if logging and log is None:
        raise AttributeError

    weights = np.zeros(shape=(X.shape[1]))
    bias = 0

    # Gradient descent is iterative since we slowly approach our optimal point
    for _epoch in range(n_iter):

        # For each data point we want to find the error and then adjust each
        # weight based on plugging that error back into a partial differential
        # of our loss function
        for i in range(y.shape[0]):

            prediction = np.dot(weights, X[i]) + bias

            # We don't need the normal squared error as it would only get used
            # in our computation later, so we can leave it "raw"
            error = prediction - y[i]

            # Taking a partial differential of the loss function gives us something
            # very similar to this, but we strip the constants for cleanliness and
            # because the user has their own control via learning_rate
            weights = weights - X[i] * learning_rate * error
            bias = bias - learning_rate * error

        if logging:
            log.append((weights, bias))

    return weights, bias
