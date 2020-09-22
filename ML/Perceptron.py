import pandas as pd, numpy as np, matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate, iterations):
        assert 0 < learning_rate < 1
        self.rate = learning_rate
        self.niter = iterations
        self.errors = []

    def fit(self, X, y):
        """
        Fit training data

        :param X Training vectors, X.shape : [#samples, #features]
        :param y Target values, y.shape : [#samples]
        """

        # weights: create a weights array of right size and
        # initialize elements to zero

        # Number of misclassifications, creates an array
        # to hold the number of misclassifications

        # main loop to fit the data to the labels

        # for i in range(self.niter):

            # set iteration error to zero
            # loop over all the objects in X and corresponding y element

            # for xi, target in zip(X, y):

                # calculate the needed (delta_w) update from prev step
                # delta_w = rate * ( target - prediction current object )

                # calculate what the current object will add to the weight

                # set the bias to be the current delta_w

                # increase the iteration error if delta_w != 0

            # Update the miscalssification array with # of errors in iteration

        # return self
        pass

    def net_input(self, X):
        """
        Calculate net input
        """
        pass

    def predict(self, X):
        """
        Return class label after unit step
        """
        pass
