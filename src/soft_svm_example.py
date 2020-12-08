import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ML.SoftSVM import SVM


def load_data():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )

    X = df.iloc[0:100, [2, 3, 4]].values
    X[:, 2] = np.where(X[:, 2] == "Iris-setosa", -1, 1)
    return X


if __name__ == "__main__":
    X = load_data()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
    ax.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor")
    ax.legend()

    classifier = SVM()
    classifier.fit(X)

    weights = classifier.weights()

    fit_x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    fit_y = (weights[0] + fit_x * weights[1]) / -weights[2]

    plt.plot(fit_x, fit_y)
    plt.fill_between(
        fit_x,
        fit_y - 2 / np.linalg.norm(weights),
        fit_y + 2 / np.linalg.norm(weights),
        edgecolor="none",
        color="#AAAAAA",
        alpha=0.4,
    )

    print(weights)

    plt.show()
