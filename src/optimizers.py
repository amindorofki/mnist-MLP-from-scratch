"""
    stochastic gradient decent optimizer implementation
"""

from typing import List

class SGD:
    "stochastic gradient decent optimizer with mini batch support"

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, layers: List) -> None:
        "updates parameters (W and b) of all layers that hold trainable weights"

        for layer in layers:
            if hasattr(layer, "W") and hasattr(layer, "dW"):
                layer.W -= self.lr * layer.dW

            if hasattr(layer, "b") and hasattr(layer, "db"):
                layer.b -= self.lr * layer.db


if __name__ == "__main__":
    import numpy as np
    from layers import Dense

    np.random.seed(10)
    layer_test = Dense(in_features = 4, out_features = 2)

    W_before = layer_test.W.copy()
    b_before = layer_test.b.copy()

    layer_test.dW = np.ones((4,2), dtype= np.float32)
    layer_test.db = np.ones((1,2), dtype= np.float32)

    optimizer = SGD(lr=0.1)
    optimizer.step([layer_test])

    print('test SGD')
    print('weight change (w_after - w_before) :\n', layer_test.W - W_before)
    print('bias change :\n', layer_test.b - b_before)

