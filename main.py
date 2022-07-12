import neural_network as nn
from activation_layer import Sigmoid
from loss_fn import MSE
from utils import fetch_mnist


def main():
    X_train, y_train = fetch_mnist(5000, normalize=True, shuffle_data=True)

    network = nn.NeuralNetwork(layers=[100, 10], input_dim=784, activation_fn=Sigmoid,
                               output_fn=Sigmoid, loss_fn=MSE)

    network.fit(X_train, y_train, learning_rate=1e-1, batch_size=64,
                test_train_ratio=0.2, epoch_num=20)


if __name__ == '__main__':
    main()
