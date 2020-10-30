import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ML.LinearRegression import linear_regression


def load_data_iris(indep_feature, dep_feature, species):
    """
    Loads data from the given csv file and allows for choice of the two
    features and two species for use in a perceptron. The arguments feature_1
    and feature_2 are mapped as follows:
        0 : Sepal Length
        1 : Sepal Width
        2 : Petal Length
        3 : Petal Width

    :param indep_feature: Integer part of the map above to be used as the independent
        feature, or the feature in the X list
    """

    if (
        all(3 < feature < 0 for feature in [indep_feature, dep_feature]) or \
        species not in ['setosa', 'versicolor', 'virginica']
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

    result_data = row_species_map[species]
    y = result_data[:, dep_feature]

    X = np.reshape(result_data[:, indep_feature], newshape=(50, -1))

    return (X, y)

if __name__ == "__main__":
    X, y = load_data_iris(0, 1, 'versicolor')
    log_var = []
    weights, bias = linear_regression(X, y, logging=True, log=log_var)
    print(weights, bias)
