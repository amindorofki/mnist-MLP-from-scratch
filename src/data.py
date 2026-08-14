"data.py - mnist data load and process"
import numpy as np
from sklearn.datasets import fetch_openml
from typing import tuple


class mnistloader:
    def __init__(self, random_state : int=1):
        self.random_state = random_state

    def onehot(self, labels:np.ndarray, num_classes: int=10) -> np.nd.array:
        num_samples = labels.shape[0]

        one_hot = np.zeros((num_samples, num_classes), dtype = np.float32)
        
        one_hot[np.arange(num_samples), labels] = 1.0


    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int64)

        X = X/255

        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]


