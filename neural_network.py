import numpy as np
from sklearn.model_selection import train_test_split

from fc_layer import FCLayer
from data_loader import DataLoader
from utils import JsonLogger

class NeuralNetwork(object):
    def __init__(self, layers: list, input_dim: int,
                 activation_fn: type, output_fn: type, loss_fn: type) -> None:
        """
        fully connected neural network, contains only dense layers, uses SGD.
        :param: layers: a list of ints representing hidden layers dimensions
        :param: input_dim: dimension of input vector
        :param: activation_fn: activation function for all hidden layers
        :param: output_fn: activation function for last layer
        :param: loss_fn: loss function provided
        """
        self.loss_fn = loss_fn()
        self.network = [FCLayer(input_dim, layers[0])]
        self.network.append(activation_fn())

        for i in range(1, len(layers)):
            self.network.append(FCLayer(layers[i - 1], layers[i]))

            if i != len(layers) - 1:
                self.network.append(activation_fn())

        self.network.append(output_fn())
        self.network_info()

    def fit(self, X: np.ndarray, y: np.ndarray, epoch_num: int = 30, batch_size: int = 1,
            learning_rate: float = 1e-3, test_train_ratio: float = 0.2, shuffle: bool = True, json_file_name: str = None) -> None:
        """
        adjusts weights to the network based on data provided, save result to json file if filename provided
        :param X: Data to fit into the model
        :param y: targets for Data (supports only categorical one-hot vectors)
        :param epoch_num: number of epochs
        :param batch_size
        :param learning_rate
        :param test_train_ratio: ratio between validation and train to be picked from overall data provided
        :param shuffle: boolean to determine whether to shuffle dataset after each epoch
        :param json_file_name: a string that represent json file name where metrics will be saved.
               if no file name was given, metrics will not be saved
        """
        # split data to train and valid datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_ratio, random_state=42)

        # create dataloaders for respective datasets
        train_dl = DataLoader(X_train, y_train, batch_size=batch_size, shuffle_data=shuffle)
        valid_dl = DataLoader(X_test, y_test, batch_size=batch_size, shuffle_data=shuffle)

        for i in range(epoch_num):
            print("Epoch", i)
            self.epoch(train_dl, "train", learning_rate, update_weights=True)
            self.epoch(valid_dl, "validation", learning_rate, update_weights=False)
            print('-' * 40)

        # save metrics to json file if file name provided (will be useful for plotting results)
        if json_file_name:
            logger = JsonLogger()
            logger.save_dict(file_name=json_file_name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predicts a given dataset
        :param X: dataset
        :return: a numpy array of predictions for data
        """
        for layer in self.network:
            X = layer.forward(X)
        return X

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        given a dataset, predicts and returns accuracy score
        :param X: data
        :param y: targets (categorical one hot vectors)
        :return: float, accuracy score for the given dataset
        """
        preds = self.predict(X)
        return self.accuracy(preds, y)

    def epoch(self, data_loader: DataLoader, epoch_type: str, learning_rate: float, update_weights: bool = True) -> None:
        """
        one complete pass of the data.
        :param data_loader: a Dataloader object that contains the data to be traversed
        :param epoch_type: a str containing the label of the epoch (for printing and json file)
        :param learning_rate
        :param update_weights: a boolean to determine whether a backpropagation is gonna occur
        """
        epoch_loss = 0
        epoch_acc = 0

        for batch_data, batch_target in data_loader:
            for x, y in zip(batch_data, batch_target):
                x, y = x.reshape(1, -1), y.reshape(1, -1)

                for layer in self.network:
                    x = layer.forward(x)

                # get metrics
                epoch_loss += self.loss_fn(x, y)
                epoch_acc += self.accuracy(x, y)

                # backward pass
                if update_weights:
                    out_error = self.loss_fn.backward(x, y)
                    for layer in reversed(self.network):
                        out_error = layer.backward(out_error)

            # update gradients for current batch
            if update_weights:
                for layer in self.network:
                    layer.update_grads(learning_rate)


        # get overall epoch metrics and print them
        epoch_loss /= data_loader.data_size
        epoch_acc /= data_loader.data_size
        print(f"{epoch_type} loss: {epoch_loss}")
        print(f"{epoch_type} acc: {epoch_acc}")

        # save current metrics (for plotting)
        JsonLogger.log[epoch_type]["loss"].append(epoch_loss)
        JsonLogger.log[epoch_type]["accuracy"].append(epoch_acc)


    def network_info(self) -> None:
        """
        prints the structure of the network
        """
        print('#' * 70)
        print('# Network Structure:')
        for i, layer in enumerate(self.network):
            print(f"# {i}) {str(layer)}")
        print('#' * 70, '\n')

    @staticmethod
    def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
        """
        accuracy metric
        :param preds: model predictions
        :param targets: ground truth for given predictions
        :return: a float between [0,1] representing the accuracy
        """
        return np.mean(np.argmax(preds, axis=1) == np.argmax(targets, axis=1))
