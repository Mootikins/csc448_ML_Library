from math import inf

import numpy as np


class DecisionStump:
    def __init__(self, epochs=20):
        self._epochs = epochs
        self._dim = 0
        self._threshold = 0
        self._minerror = inf
        self._ineq = ""

    def classify(self, X, dim, threshold, inequality):
        return_arr = np.ones((np.shape(X)[0], 1))
        if inequality == "lt":
            return_arr[X[:, dim] <= threshold] = -1.0
        else:
            return_arr[X[:, dim] > threshold] = -1.0
        return return_arr

    def fit(self, X, y, D=None):
        if D is None:
            D = np.mat(np.ones((len(X), 1)) / len(X))
        else:
            assert len(D) == len(X)

        data = np.mat(X)
        labels = np.mat(y).T
        m, n = np.shape(data)
        self._minerror = inf
        for i in range(n):

            rangeMin = data[:, i].min()
            rangeMax = data[:, i].max()
            stepSize = (rangeMax - rangeMin) / self._epochs

            for j in range(-1, int(self._epochs) + 1):
                for inequal in ["lt", "gt"]:
                    threshold = rangeMin + float(j) * stepSize
                    predictions = self.classify(data, i, threshold, inequal)
                    errArr = np.mat(np.ones((m, 1)))
                    errArr[predictions == labels] = 0
                    weighted_err = D.T * errArr
                    if weighted_err < self._minerror:
                        self._minerror = weighted_err
                        self._dim = i
                        self._threshold = threshold
                        self._ineq = inequal

    def get_threshold(self):
        """
        :return: The fit threshold
        """
        return self._threshold

    def get_dimension(self):
        """
        :return: the fit dimension
        """
        return self._dim
