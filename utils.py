import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import os
import glob

class JsonLogger:
    """
    a simple "logging" class, to save model metrics in json format. json file will be used to plot results.
    :param: log: a static data member in the form of dictionary that will be converted to json file when saved
    """
    # initiate dict
    log = {"train": {}, "validation": {}}
    log["train"]["loss"], log["train"]["accuracy"] = [], []
    log["validation"]["loss"], log["validation"]["accuracy"] = [], []

    def save_dict(self, file_name: str, dest: str = '/content/res', ) -> None:
        with open(f'/content/res/{file_name}.json', 'w+') as fp:
            json.dump(self.log, fp, indent=2)
        self.reset()

    def reset(self):
        JsonLogger.log.clear()
        JsonLogger.log = {"train": {}, "validation": {}}
        JsonLogger.log["train"]["loss"], JsonLogger.log["train"]["accuracy"] = [], []
        JsonLogger.log["validation"]["loss"], JsonLogger.log["validation"]["accuracy"] = [], []

def plot_exp_results(filename_pattern, results_dir='res') -> None:
    """
    a function to plot all json files of the experiment
    :param filename_pattern: the suffix of all json files related to current experiment, of template "exp*.json"
    :param results_dir: directory where files are located
    """
    figure, axis = plt.subplots(2, 2)
    figure.tight_layout()
    result_files = glob.glob(os.path.join(results_dir, filename_pattern))
    result_files.sort()

    for filepath in result_files:
        basename = os.path.basename(filepath)
        match = re.match('exp\d_(\d_)?(.*)\.json', basename)
        if match:
            with open(filepath) as file:
                experiment = json.load(file)
                axis[0, 0].plot(experiment["train"]["loss"], label=basename), axis[0, 0].title.set_text("train loss")
                axis[0, 1].plot(experiment["train"]["accuracy"], label=basename), axis[0, 1].title.set_text("train accuracy")
                axis[1, 1].plot(experiment["validation"]["accuracy"], label=basename), axis[1, 1].title.set_text("validation accuracy")
                axis[1, 0].plot(experiment["validation"]["loss"], label=basename), axis[1, 0].title.set_text("validation loss")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def fetch_mnist(sample_num: int = 60000, normalize: bool = True, shuffle_data: bool = True) -> tuple:
    """
    fetch mnist from csv file according to flags provided
    :param sample_num: number of samples to be taken from dataset (takes first n samples)
    :param normalize: divide by 255
    :param shuffle_data: boolean to determine if shuffles data before taking first n samples
    :return: numpy arrays of data and targets of the MNIST dataset
    """
    data = pd.read_csv("mnist_train.csv")

    if shuffle_data:
        np.random.shuffle(data.values)

    # get target and transform to one-hot vector
    target = data.pop("label").to_numpy(dtype=int)
    target = np.eye(10)[target][:sample_num, :]

    # get subset of data, and normalize
    data = data.to_numpy(dtype="float32")[:sample_num, :]

    if normalize:
        data /= 255


    return data, target