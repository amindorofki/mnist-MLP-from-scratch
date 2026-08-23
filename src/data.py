"data.py - mnist data load and process"
import numpy as np
from sklearn.datasets import fetch_openml
from typing import Tuple


class MNISTloader:
    def __init__(self, random_state : int=1):
        self.random_state = random_state

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int64)

        X = X/255

        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        
        y_train_oh = self.onehot(y_train)
        y_test_oh = self.onehot(y_test)

        print(f"""data loaded\n
        x_train shape : {X_train.shape}\n
        y_train shape : {y_train.shape}\n
        y_onehot shape : {y_train_oh.shape}""")

        return X_train, y_train_oh, y_train, X_test, y_test_oh, y_test

    def onehot(self, labels:np.ndarray, num_classes: int=10) -> np.ndarray:
        num_samples = labels.shape[0]

        one_hot = np.zeros((num_samples, num_classes), dtype = np.float32)
        
        one_hot[np.arange(num_samples), labels] = 1.0

        return one_hot

def train_val_split(X: np.ndarray, Y: np.ndarray, val_ratio: float=0.1, seed: int=10):
        "split training set into train and validation sets"
        rng = np.random.RandomState(seed)
        n = X.shape[0]
        indices = rng.permutation(n)
        valz_size = int(n * val_ratio)
        val_idx, train_idx = indices[:valz_size], indices[valz_size:]
        return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

if __name__ == "__main__":
    loader = MNISTloader()
    X_train, y_train_oh, y_train, X_test, y_test_oh, y_test = loader.load_data()
