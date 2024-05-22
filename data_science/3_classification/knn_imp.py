import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
import pandas as pd

class KNNTO:

    def __init__(self, k=3) -> None:
        self.k=k

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y, pd.DataFrame):
            y = y.values.flatten()
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(list(k_nearest_labels)).most_common(1)
        return most_common[0][0]
    


def main():
    knnto = KNNTO()
    iris = load_iris()
    print(iris)



if __name__ == "__main__":
    main()


