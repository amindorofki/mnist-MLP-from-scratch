"""
    fully connected layer implementation
"""

import numpy as np

class Dense:
    "fully connected Dense layer"

    def __init__(self, in_features: int, out_features: int):
        "initializing weights"
        self.in_features = in_features
        self.out_features = out_features

        self.W = np.random.randn(
            in_features, 
            out_features) * np.sqrt(2/in_features)

        self.b = np.zeros((1, out_features), dtype= np.float32)

        self.X = None

        self.dW = None
        self.db = None

    def forward(self, X:np.ndarray) -> np.ndarray:
        "compute affine transformation Z = XW + B"
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ:np.ndarray, l2_lambda: float=0.0) -> np.ndarray:
        "computes gradients for w and b and propagates error to previous layer"

        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)
        
        self.dW += l2_lambda * self.W 
        return dX


if __name__ == '__main__':
    np.random.seed(10)
    dense = Dense(in_features=784, out_features=128)

    x_test = np.random.randn(2, 784)
    dZ_test = np.random.randn(2, 128)

    Z_test = dense.forward(x_test)

    dX_test = dense.backward(dZ_test)

    print("test dense layer")
    print("forward Z shape :", Z_test.shape)
    print("backward dX shape :", dX_test.shape)
    print("gradient dW shape :", dense.dW.shape)
    print("gradient db shape :", dense.db.shape)


