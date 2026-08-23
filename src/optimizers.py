"""
    stochastic gradient decent optimizer implementation
"""

from typing import List
import numpy as np 


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

class Momentum:
    "SGD with Momentum smooths gradient updates using a moving average"

    def __init__(self, lr: float= 0.01, beta : float=0.9):
        self.lr = lr 
        self.beta = beta 
        self.v = {} # velocity

    def step(self, layers):
        for layer in layers:
            key = id(layer)
            if key not in self.v:
                self.v[key] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

            self.v[key]["W"] = self.beta * self.v[key]["W"] + (1 - self.beta) * layer.dW
            self.v[key]["b"] = self.beta * self.v[key]["b"] + (1 - self.beta) * layer.db 

            layer.W -= self.lr * self.v[key]["W"]
            layer.b -= self.lr * self.v[key]["b"]


class Adam:
    "Adam optimizer per-parameter adaptive learning rates"
    
    def __init__(self, lr : float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float= 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
        
    def step(self, layers):
        self.t += 1
        for layer in layers:
            key = id(layer)
            if key not in self.m:
                self.m[key] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
                self.v[key] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
            for p, grad in (("W", layer.dW), ("b", layer.db)):
                self.m[key][p] = self.beta1 * self.m[key][p] + (1 - self.beta1) *grad
                self.v[key][p] = self.beta2 * self.v[key][p] + (1 - self.beta2) * (grad **2)
                
                m_hat = self.m[key][p] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key][p] / (1 - self.beta2 ** self.t)

                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                if p == "W":
                    layer.W -= update
                else:
                    layer.b -= update

if __name__ == "__main__":
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

