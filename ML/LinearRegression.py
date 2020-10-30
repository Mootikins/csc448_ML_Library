import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def LinearRegression(X, y, n_iter=500, learning_rate=0.01):
    """
    Fit the linear regression for input variable X to output variable y

    :param X: ndarray of floats of shape (num_features, num_data_points)
    :param y: ndarray of floats of shape (num_data_points, 1)
    :param n_iter: int, number of iterations to perform given it does not
        correctly approach the bottom
    :param learning_rate: float, should be between 0 and 1. Determines how
        "aggressive" the weight gradient application is

    :return weights as a numpy array and the bias. If the input data was one
        dimensional, then the weight is the slope and the bias is the y-intercept
    """
    weights = np.zeros(shape=(X.shape[1]))
    bias = 0

    # Gradient descent is iterative since we slowly approach our optimal point
    for _epoch in range(n_iter):
        # For each data point we want to find the error and then adjust each
        # weight based on plugging that error back into a partial differential
        # of our loss function
        for i in range(y.shape[0]):
            # Build our prediction based on our weights
            prediction = np.dot(weights, X[i]) + bias
            # We don't need the normal squared error so we just use this
            error = prediction - y[i]
            weights = weights - X[i] * learning_rate * error
            bias = bias - learning_rate * error

    return weights, bias
