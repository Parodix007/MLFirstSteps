# Supervised Machine Learning Algorithms
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plo
from sklearn.neighbors import KNeighborsClassifier
# * *  Simple iris species machine learning
dataset = load_iris()


X_train, X_test, y_train, y_test = train_test_split(
        dataset['data'], dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new.shape)

prediction = knn.predict(X_new)
print(prediction)
print(f'prediction name {dataset["target_names"][prediction]}')

y_pred = knn.predict(X_test)

print(y_pred)

print(f'test score: {np.mean(y_pred == y_test)}')

print("--------------")
# * KNeighbors from sklearn

bc = load_breast_cancer()

bcx_train, bcx_test, bcy_train, bcy_test = train_test_split(bc.data, bc.target, random_state=0)

train_acc = []
test_acc = []
number_of_n = range(1, 101)
for n in number_of_n:
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(bcx_train, bcy_train)
        train_acc.append(model.score(bcx_train, bcy_train))
        test_acc.append(model.score(bcx_test, bcy_test))

# Show acc of train set and test set
plo.plot(number_of_n, train_acc, label="train acc")
plo.plot(number_of_n, test_acc, label="test acc")
plo.ylabel("Accuracy")
plo.xlabel("n_neighbors")
plo.legend()
plo.show()
