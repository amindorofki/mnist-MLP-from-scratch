"""
    gradient_check.py - verifies backward pass correctness via numerical differentiation
"""

import numpy as np 
from src.layers import Dense 
from src.activations import ReLU, Softmax
from src.losses import CrossEntropy

def numerical_gradient(param, X, Y, forward_fn, loss_fn, eps =1e-5):
    "estimates dL/dparam via central difference, entry by entry"
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orginal_value = param[idx]

        param[idx] = orginal_value + eps
        loss_plus = loss_fn.forward(forward_fn(X), Y)
        
        param[idx] = orginal_value - eps
        loss_minus = loss_fn.forward(forward_fn(X), Y)
    
        param[idx] = orginal_value
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad

def relative_error(analytic, numeric):
    return np.abs(analytic - numeric) / np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))

def check_gradients():
    np.random.seed(1)

    dense1 = Dense(in_features = 4, out_features = 5)
    relu = ReLU()
    dense2 = Dense(in_features = 5, out_features =3)
    softmax = Softmax()
    loss_fn = CrossEntropy()

    X = np.random.randn(3, 4)
    Y = np.zeros((3, 3))
    Y[np.arange(3), np.random.randint(0, 3, size=3)] =1

    def forward_fn(X):
        Z1 = dense1.forward(X)
        A1 = relu.forward(Z1)
        Z2 = dense2.forward(A1)
        return softmax.forward(Z2)

    # ---- analytic gradients --- 
    A2 = forward_fn(X)
    loss_fn.forward(A2, Y)
    dZ2 = loss_fn.backward()
    dA1 = dense2.backward(dZ2)
    dZ1 = relu.backward(dA1)
    _ = dense1.backward(dZ1)

    params = {
        "dense1.W": (dense1.W, dense1.dW),
        "dense1.b": (dense1.b, dense1.db),
        "dense2.W": (dense2.W, dense2.dW),
        "dense2.b": (dense2.b, dense2.db),
    }

    print("=== Gradient check ===")
    for name, (param, analytic_grad) in params.items():
        numeric_grad = numerical_gradient(param, X, Y, forward_fn, loss_fn)
        err = np.max(relative_error(analytic_grad, numeric_grad))
        status = 'PASS' if err < 1e-4 else "FAIL"
        print(f"{name:12s} | max relative error = {err:.2e} | [{status}]")

if __name__ == "__main__":
    check_gradients()
