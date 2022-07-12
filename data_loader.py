import numpy as np
from sklearn.utils import shuffle


class DataLoader(object):
    def __init__(self, data: np.ndarray, target: np.ndarray, batch_size: int = 64, shuffle_data=True) -> None:
        """
        an iterable dataloader object that yields mini-batches of data at the specified batch size
        :param shuffle_data: whether to shuffle data upon initialization
        """
        self.data = data
        self.shuffle_data = shuffle_data
        self.target = target
        self.data_size = target.shape[0]
        self.batch_size = batch_size
        self.cur_batch_index = 0

    def __iter__(self) -> tuple:
        # shuffle data each pass through data (each epoch)
        if self.shuffle_data:
            self.data, self.target = shuffle(self.data, self.target)

        while self.cur_batch_index < self.data_size:
            # check if last batch (might be smaller then a regular batch)
            if self.data_size - self.cur_batch_index < self.batch_size:
                cur_batch_size = self.data_size - self.cur_batch_index
            else:
                cur_batch_size = self.batch_size

            # generate batch data and target
            yield self.data[self.cur_batch_index: self.cur_batch_index + cur_batch_size, :], \
                  self.target[self.cur_batch_index:self.cur_batch_index + cur_batch_size]

            self.cur_batch_index += cur_batch_size

        # reset index back to zero upon completion
        self.cur_batch_index = 0
