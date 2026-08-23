"""
    model.py - sequential container for chaining layers into a full network
"""

from typing import List

class Sequential:
    "chains a list of layers into a full forward/backward pipeline"

    def __init__(self, layers: List):
        self.layers = layers

    def forward(self, X):
        "runs x through every layer in order"
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, dout, l2_lambda:float=0.0):
        "propagates gradient backward through every layer in reverse order"
        grad = dout
        for layer in reversed(self.layers):
            if hasattr(layer, "W"):
                grad = layer.backward(grad, l2_lambda=l2_lambda)
            else: 
                grad = layer.backward(grad)
        return grad

    def trainable_layers(self):
        "returns only layers that hold weights (for the optimizer)"
        return [layer for layer in self.layers if hasattr(layer, "W")]
