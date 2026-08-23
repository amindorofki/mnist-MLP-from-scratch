"unit tests for ReLU and Softmax activations"
import numpy as np
from src.activations import ReLU, Softmax


def test_relu_zeroes_negative_values():
    relu = ReLU()
    z = np.array([[-2.0, 0.0, 3.0]])
    out = relu.forward(z)
    np.testing.assert_array_equal(out, [[0.0, 0.0, 3.0]])


def test_relu_backward_blocks_gradient_where_input_was_negative():
    relu = ReLU()
    z = np.array([[-1.0, 2.0]])
    relu.forward(z)
    dA = np.array([[1.0, 1.0]])
    dZ = relu.backward(dA)
    np.testing.assert_array_equal(dZ, [[0.0, 1.0]])


def test_softmax_output_sums_to_one():
    softmax = Softmax()
    logits = np.array([[2.0, 1.0, 0.1]])
    probs = softmax.forward(logits)
    assert np.isclose(np.sum(probs), 1.0)


def test_softmax_output_is_nonnegative():
    softmax = Softmax()
    logits = np.array([[5.0, -3.0, 0.0]])
    probs = softmax.forward(logits)
    assert np.all(probs >= 0)