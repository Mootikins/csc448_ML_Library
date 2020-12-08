import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ML.LogisticRegression import LogisticRegression

if __name__ == "__main__":
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )

    X = df.iloc[0:100, 2].values

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 0, 1)

    plt.scatter(X[:50], y[:50], color="red", marker="o")
    plt.scatter(X[50:], y[50:], color="blue", marker="x")
    log_regression = LogisticRegression()
    logger = []
    log_regression.fit(X, y, logger)
    # print(logger)

    fit_x = np.linspace(np.min(X[:]), np.max(X[:]), 100)
    fit_y = log_regression.predict(fit_x)

    plt.plot(fit_x, fit_y)
    plt.show()
