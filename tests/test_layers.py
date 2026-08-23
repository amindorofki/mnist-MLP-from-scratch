"unit tests for Dense layer: shape correctness and forward/backward consistency"
import numpy as np
from src.layers import Dense


def test_dense_forward_output_shape():
    "Dense with in=4, out=3 should map a (batch, 4) input to (batch, 3) output"
    layer = Dense(in_features=4, out_features=3)
    X = np.random.randn(5, 4)
    out = layer.forward(X)
    assert out.shape == (5, 3)


def test_dense_backward_output_shape():
    "backward should return a gradient matching the ORIGINAL input shape"
    layer = Dense(in_features=4, out_features=3)
    X = np.random.randn(5, 4)
    layer.forward(X)
    dOut = np.random.randn(5, 3)
    dX = layer.backward(dOut)
    assert dX.shape == (5, 4)


def test_dense_weight_gradient_shape():
    "dW must match W's shape, db must match b's shape"
    layer = Dense(in_features=4, out_features=3)
    X = np.random.randn(5, 4)
    layer.forward(X)
    layer.backward(np.random.randn(5, 3))
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape