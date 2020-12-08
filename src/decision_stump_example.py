import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ML.DecisionStump import DecisionStump


def load_data():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )

    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y[:] == "Iris-setosa", -1, 1)
    return X, y


if __name__ == "__main__":
    X, y = load_data()

    stump = DecisionStump()

    stump.fit(X, y)

    fit_x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    fit_y = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

    plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o")
    plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x")

    # Dimension is which coordinate the split is occuring on,
    # so we map it across the x coor and y coor
    if stump.get_dimension() == 0:
        fit_x[:] = stump.get_threshold()
    else:
        fit_y[:] = stump.get_threshold()

    plt.plot(fit_x, fit_y)

    plt.show()
