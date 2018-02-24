"""General utility functions"""

import json
import logging
import os
import glob
import numpy as np
import os

def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def creatLabelIndex(labelMask,label):
    labelIndex = np.zeros(label.shape)
    for i , ulabel in enumerate(labelMask):
        labelIndex[np.where(label==ulabel)]=i
    return labelIndex

def creatLabelMask(label):
    uniquelabel = np.unique(label)
    return uniquelabel

def datatofileArrlable(datadir):
    labels = os.listdir(datadir)
    labelArr = [] 
    soundArr = []
    for label in labels:
        fileNames = glob.glob(os.path.join(datadir,label)+"/*.wav")
        for name in fileNames: 
            labelArr.append(label)
            soundArr.append(name)
    return np.array(labelArr), np.array(soundArr)
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.astype(int).reshape(-1)].T
    return Y

def createfeedDict(inputs,minibatch,feeddatadict=None):
    feed_dict = {inputs['images']:minibatch[0], inputs['labels']:minibatch[1]} 
    if feeddatadict is not None:
        for keys in feeddatadict.keys():
            feed_dict[keys] = feeddatadict[keys]
    return feed_dict
