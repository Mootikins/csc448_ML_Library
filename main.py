import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ML.Perceptron import Perceptron
from ML.Perceptron import plot_decision_regions


def load_data(features, species_1, species_2):
    """
    Loads data from the given csv file and allows for choice of the two
    features and two species for use in a perceptron. The arguments feature_1
    and feature_2 are mapped as follows:
        0 : Sepal Length
        1 : Sepal Width
        2 : Petal Length
        3 : Petal Width

    :param features List of Integers: Unique list of integers in range 0-3 to
                                      define which feature to use; mapping is
                                      shown above
    :param species_1 String: String defining the first species of flower to use;
                           must be one of ['setosa', 'versicolor', 'virginica']
                           and different from species_2
    :param species_2 String: String defining the first species of flower to use;
                           must be one of ['setosa', 'versicolor', 'virginica']
                           and different from species_1
    """

    if (
        all(3 < feature < 0 for feature in features) or \
        len(features) != len(set(features)) or \
        species_1 not in ['setosa', 'versicolor', 'virginica'] or \
        species_2 not in ['setosa', 'versicolor', 'virginica']
    ):
        print("Check input for `load_data`")
        sys.exit(1)

    data_frame = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None)

    row_species_map = {
        'setosa': data_frame.iloc[:50].values,
        'versicolor': data_frame.iloc[50:100].values,
        'virginica': data_frame.iloc[100:].values,
    }

    result_data = np.concatenate(
        (row_species_map[species_1], row_species_map[species_2]))
    y = result_data[:, 4]
    y = np.where(y == 'Iris-' + species_1, -1, 1)

    X = result_data[:, features]

    return (X, y)


def show_plot(X, data_1_label, data_2_label, x_label, y_label):
    plt.scatter(X[:50, 0],
                X[:50, 1],
                color='red',
                marker='o',
                label=data_1_label)

    plt.scatter(X[50:100, 0],
                X[50:100, 1],
                color='blue',
                marker='x',
                label=data_2_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    PN = Perceptron(0.1, 10)
    # feature number 0 is sepal length, feature number 2 is petal length
    (X, y) = load_data([0, 2], species_1='setosa', species_2='virginica')

    PN.fit(X, y)
    print(PN.misclassifications)

    plt = plot_decision_regions(X, y, PN)
    plt.show()

    # show_plot(X, 'setosa', 'virginica', 'sepal length', 'petal length')
