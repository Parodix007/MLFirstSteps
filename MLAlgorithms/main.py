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

sc = pd.plotting.scatter_matrix(data_frame, alpha=0.3, figsize=(10, 10), diagonal="hist", color=colors, marker="o", grid=True)

plt.show()