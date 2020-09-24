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
        self.weights = [0] * len(y)

        # Number of misclassifications, creates an array
        # to hold the number of misclassifications
        self.misclassifications = []

        # main loop to fit the data to the labels
        for _i in range(self.niter):

            # set iteration error to zero
            misclassifified = 0
            # loop over all the objects in X and corresponding y element
            for xi, target in zip(X, y):
                x = np.insert(xi, 0, 1)
                y = np.dot(self.weights, x.transpose())
                target = 1.0 if (y > 0) else 0.0

                # calculate the needed (delta_w) update from prev step
                delta_w = target - y

                # calculate what the current object will add to the weight

                # set the bias to be the current delta_w

                # increase the iteration error if delta_w != 0
                if delta_w > 0:
                    misclassifified += 1
                    self.weights += (delta_w * x)

            # Update the miscalssification array with # of errors in iteration
            if misclassifified == 0:
                break
            else:
                self.misclassifications.append(misclassifified)

        return self

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.weights) + self.rate

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
