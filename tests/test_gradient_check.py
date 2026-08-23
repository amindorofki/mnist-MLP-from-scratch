"promotes gradient_check.py's logic into an automated pytest check"
import numpy as np
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CrossEntropy
from gradient_check import numerical_gradient, relative_error


def test_backward_pass_matches_numerical_gradient():
    np.random.seed(1)

    dense1 = Dense(in_features=4, out_features=5)
    relu = ReLU()
    dense2 = Dense(in_features=5, out_features=3)
    softmax = Softmax()
    loss_fn = CrossEntropy()

    X = np.random.randn(3, 4)
    Y = np.zeros((3, 3))
    Y[np.arange(3), np.random.randint(0, 3, size=3)] = 1

    def forward_fn(X):
        Z1 = dense1.forward(X)
        A1 = relu.forward(Z1)
        Z2 = dense2.forward(A1)
        return softmax.forward(Z2)

    A2 = forward_fn(X)
    loss_fn.forward(A2, Y)
    dZ2 = loss_fn.backward()
    dA1 = dense2.backward(dZ2)
    dZ1 = relu.backward(dA1)
    dense1.backward(dZ1)

    params = {
        "dense1.W": (dense1.W, dense1.dW),
        "dense1.b": (dense1.b, dense1.db),
        "dense2.W": (dense2.W, dense2.dW),
        "dense2.b": (dense2.b, dense2.db),
    }

    for name, (param, analytic_grad) in params.items():
        numeric_grad = numerical_gradient(param, X, Y, forward_fn, loss_fn)
        err = np.max(relative_error(analytic_grad, numeric_grad))
        assert err < 1e-4, f"{name} failed gradient check with error {err:.2e}"