"""
    categorical Cross-enthropy loss fuction and gradient
"""

import numpy as np

class Crossenthropy:
    def __init__(self, eps:float = 1e-15):
        self.eps = eps
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true

        y_pred_clip = np.clip(y_pred, self.eps, 1 - self.eps)

        loss = -np.mean(np.sum(y_true * np.log(y_pred_clip), axis = -1))
        return float(loss)

    def backward(self) -> np.ndarray:
        N = self.y_true.shape[0]
        return (self.y_pred - self.y_true)/ N 


if __name__ == '__main__':
    # TEST crossenthropy
    ce = Crossenthropy()

    # test batch of 2 samples, 3 classes 

    y_true_test = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
    y_pred_test = np.array(
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], dtype = np.float32
    )

    loss_val = ce.forward(y_pred_test, y_true_test)
    grad_val = ce.backward()

    print('test cross enthory loss')
    print('loss : ', loss_val)
    print('gradient dz shape : ', grad_val.shape)
    print('gradient dz : ', grad_val)

