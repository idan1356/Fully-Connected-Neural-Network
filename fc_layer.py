import numpy as np


class FCLayer(object):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        fully-connected layer of the network
        """
        self.in_dim = input_dim
        self.out_dim = output_dim

        norm = np.sqrt(input_dim + output_dim)
        self.weights = np.random.randn(input_dim, output_dim) / norm
        self.bias = np.random.randn(1, output_dim) / norm
        self.input = None


        self.nabla_w = np.zeros((input_dim, output_dim))
        self.nabla_b = np.zeros((1, output_dim))
        self.batch_size = 0

    def __str__(self) -> str:
        shape = self.weights.shape
        return f"{self.__class__} (In_Dim: {shape[0]}, Out_Dim: {shape[1]})"

    def forward(self, input_arr: np.ndarray) -> np.ndarray:
        self.input = input_arr
        return np.dot(input_arr, self.weights) + self.bias

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        # calculate gradients
        d_w = np.dot(self.input.T, d_y)
        d_x = np.dot(d_y, self.weights.T)

        # accumulate mini-batch gradients
        self.nabla_w += d_w
        self.nabla_b += d_y
        self.batch_size += 1

        """# update weights and biases
        self.weights -= learning_rate * d_w
        self.bias -= learning_rate * d_y"""
        return d_x

    def update_grads(self, learning_rate):
        # update weights by mean of accumulated gradients
        self.weights -= (learning_rate / self.batch_size) * self.nabla_w
        self.bias -= (learning_rate / self.batch_size) * self.nabla_b

        # reset
        self.nabla_w = np.zeros((self.in_dim, self.out_dim))
        self.nabla_b = np.zeros((1, self.out_dim))
        self.batch_size = 0

