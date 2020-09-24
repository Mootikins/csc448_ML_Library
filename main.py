import pandas as pd, numpy as np, matplotlib.pyplot as plt
from ML.Perceptron import Perceptron


def load_data():
    data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None)
    y = data.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = data.iloc[0:100, [0, 2]].values

    return (y, X)


def show_plot(X, y):
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')

    plt.scatter(X[50:100, 0],
                X[50:100, 1],
                color='blue',
                marker='x',
                label='versicolor')

    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    # PN = Perceptron(0.1, 10)
    (y, X) = load_data()

    show_plot(X, y)
