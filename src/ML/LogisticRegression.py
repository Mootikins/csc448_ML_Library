import numpy as np


class LogisticRegression:
    def __init__(self, epochs=100, learning_rate=0.5):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._weights = np.zeros((1, 1))

    def predict(self, row):
        return 1.0 / (1.0 + np.exp(-(self._weights[0] + self._weights[1] * row)))

    def fit(self, X, y, logger=None):
        try:
            self._weights = [0.0 for i in range(X.shape[1] + 1)]
        except IndexError:
            self._weights = [0.0 for i in range(2)]

        for epoch in range(self._epochs):
            sum_error = 0
            for i in range(X.shape[0]):
                row = X[i]
                yhat = self.predict(row)
                error = y[i] - yhat
                sum_error += error ** 2
                self._weights[0] = self._weights[0] + self._learning_rate * error * yhat * (
                    1.0 - yhat
                )
                self._weights[1] = (
                    self._weights[1]
                    + self._learning_rate * error * yhat * (1.0 - yhat) * row
                )
                if sum_error == 0:
                    break

            if sum_error == 0:
                break

            if logger is not None:
                logger += (epoch, self._learning_rate, sum_error)

    def get_coeff(self):
        return self._weights
