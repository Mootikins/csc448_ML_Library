import pandas as pd, numpy as np, matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate, iterations):
        assert 0 < learning_rate < 1
        self.learning_rate = learning_rate
        self.iterations = iterations

    def print_vals(self):
        print(self.learning_rate, self.iterations)

    def fit(self):
        pass
