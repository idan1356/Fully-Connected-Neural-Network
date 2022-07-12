import numpy as np
from abc import ABC, abstractmethod


class ActivationLayer(ABC):
    def __init__(self):
        """
        abstract class for the activation layer of the network
        """
        self.input = None
        self.output = None

    def __str__(self) -> str:
        return str(self.__class__)

    @abstractmethod
    def activate(self, input_arr: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, input_arr: np.ndarray) -> np.ndarray:
        pass

    def forward(self, input_arr: np.ndarray) -> np.ndarray:
        self.input = input_arr
        return self.activate(input_arr)

    def backward(self, d_y: np.ndarray, *args) -> np.ndarray:
        return self.derivative(self.input) * d_y

    def update_grads(self, *args):
        pass

class Sigmoid(ActivationLayer):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig_x = self.activate(x)
        return sig_x * (1 - sig_x)


class ReLu(ActivationLayer):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1. * (x > 0)

