import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


class Perceptron:
    def __init__(self, learning_rate, iterations):
        assert 0 < learning_rate < 1
        self.learning_rate = learning_rate
        self.niter = iterations
        self.errors = []
        self.bias = 1

    def fit(self, X, y):
        """
        Fit training data

        :param X Training vectors, X.shape : [#samples, #features]
        :param y Target values, y.shape : [#samples]
        """

        # weights: create a weights array of right size and
        # initialize elements to zero
        self.weights = [0] * len(X[0])

        # insert our constant into the beginning of the input array
        for xi in X:
            xi = np.insert(xi, 0, 1)

        # Number of misclassifications, creates an array
        # to hold the number of misclassifications
        self.misclassifications = []

        # main loop to fit the data to the labels
        for _i in range(self.niter):

            # set iteration error to zero
            misclassifified = 0
            # loop over all the objects in X and corresponding y element
            for xi, target in zip(X, y):
                prediction = self.predict(xi)

                # calculate the needed (delta_w) update from prev step
                delta_w = self.learning_rate * (target - prediction)

                # calculate what the current object will add to the weight

                # set the bias to be the current delta_w
                self.bias = delta_w

                # increase the iteration error if delta_w != 0
                if delta_w != 0:
                    misclassifified += 1
                    self.weights += (delta_w * xi)

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
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    return plt
