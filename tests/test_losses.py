"unit tests for CrossEntropy loss"
import numpy as np
from src.losses import CrossEntropy


def test_cross_entropy_zero_when_prediction_is_perfect():
    "if the model predicts the true class with prob ~1, loss should be ~0"
    loss_fn = CrossEntropy()
    probs = np.array([[0.999, 0.0005, 0.0005]])
    Y = np.array([[1, 0, 0]])
    loss = loss_fn.forward(probs, Y)
    assert loss < 0.01


def test_cross_entropy_high_when_prediction_is_wrong():
    "if the model confidently predicts the WRONG class, loss should be large"
    loss_fn = CrossEntropy()
    probs = np.array([[0.001, 0.998, 0.001]])
    Y = np.array([[1, 0, 0]])
    loss = loss_fn.forward(probs, Y)
    assert loss > 5.0