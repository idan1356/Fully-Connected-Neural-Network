import numpy as np
from abc import ABC


class LossFunction(ABC):
    def __init__(self):
        """
        abstract class for the loss function of the network, includes both forward and backward propagations
        """
        self.output = None

    def __call__(self, preds: np.ndarray, target: np.ndarray) -> float:
        return self.forward(preds, target)

    def forward(self, preds: np.ndarray, target: np.ndarray) -> float:
        pass

    def backward(self, preds: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class MSE(LossFunction):
    def forward(self, preds: np.ndarray, target: np.ndarray) -> float:
        return np.mean(np.power(target - preds, 2))

    def backward(self, preds: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2 * (preds - target) / target.size
