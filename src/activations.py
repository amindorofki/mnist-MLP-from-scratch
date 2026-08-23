"""
    activation functions and their derivatives
"""

import numpy as np

class ReLU:
    "Rectified linear unit activation function"

    def __init__(self):
        self.z = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        "computes the RelU activation: f(z) = max(0, z)"

        self.z = z
        return np.maximum(0, z)

    def backward(self, dA: np.ndarray = None) -> np.ndarray:
        "computes backward pass for ReLU"
        
        return dA * (self.z > 0).astype(np.float32)

class Softmax:
    "softmax activation function"

    def __init__(self):
        self.z = None
        self.out = None

    def forward(self, z:np.ndarray) -> np.ndarray:
        self.z = z 
        shift_z = z - np.max(z, axis = -1, keepdims=True)
        exps = np.exp(shift_z)
        prob = exps/np.sum(exps, axis = -1, keepdims =True)
        self.out = prob
        return prob

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        return dZ


if __name__ == "__main__":
    #test ReLU
    relu = ReLU()
    z_test = np.array([[-2.0, 0.0, 3.0]], dtype = np.float32)
    dA_test = np.array([[1.0, 1.0, 1.0]])
    print("test relu")
    print("test input", z_test)
    print("forward : ", relu.forward(z_test) )
    print("derivatives : ", relu.backward(dA_test))

    # test softmax
    softmax = Softmax()
    logits_test = np.array([2.0, 1.0, 0.1])
    probs = softmax.forward(logits_test)
    print("test softmax")
    print("probabilities : ", probs)
    print("row sum (must be 1): ", np.sum(probs ))

