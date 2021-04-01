from IPython.core.pylabtools import figsize
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
colors = list()

colors_opt = {0: "red", 1: "green", 2: "blue"}

for n in np.nditer(iris.target):
    colors.append(colors_opt[int(n)])

data_frame = pd.DataFrame(iris.data, columns=iris.feature_names)

#sc = pd.plotting.scatter_matrix(data_frame, alpha=0.3, figsize=(10, 10), diagonal="hist", color=colors, marker="o", grid=True)

#plt.show()


# Perceptron Algorithm

class Perceptron:
    """
        It will be tested with Iris dataset from scikit-learn
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
            return self

    def net_input(self,X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


data_frame.tail()



