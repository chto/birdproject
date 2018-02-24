"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.training import train_and_evaluate
import glob 
import numpy as np
import model.utils as util
import model.input_fn as input_fn
import model.model_fn as model_fn
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--noise_dir', default='data/64x64_SIGNS',
                    help="Directory containing the noise data")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    #model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    model_dir_has_best_weights=False
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"
    try:
        fileName = os.path.join(args.model_dir, params.transferdata)
        with open(fileName, 'rb') as f:
            data = pickle.load(f)
        logging.info("Reading transfer learning dataset")
    except:
        print("None transfer")
        data = None
    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train/")
    dev_data_dir = os.path.join(data_dir, "valid/")

    # Get the filenames from the train and dev sets
    noise_filenames = np.array(glob.glob(os.path.join(args.noise_dir, "*.wav")))
    train_labels, train_filenames =  util.datatofileArrlable(train_data_dir)
    eval_labels, eval_filenames = util.datatofileArrlable(dev_data_dir)
    LabelMask = util.creatLabelMask(train_labels) 
    train_labels_index = util.creatLabelIndex(LabelMask,train_labels)
    eval_labels_index = util.creatLabelIndex(LabelMask,eval_labels)
    train_labels_index = util.convert_to_one_hot(train_labels_index, len(LabelMask)).T
    eval_labels_index = util.convert_to_one_hot(eval_labels_index, len(LabelMask)).T
    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)
    print("training_size:{0}, eval_size{1}".format(params.train_size, params.eval_size))
    # Create the two iterators over the two datasets
    train_inputs = input_fn.Batchiterator(train_filenames, train_labels_index, noise_filenames, params)
    eval_inputs = input_fn.Batchiterator(eval_filenames, eval_labels_index, None, params, is_eval=True)
    
    images =  tf.placeholder(tf.float32, shape=(None,228, 517,1), name="images")
    labels =  tf.placeholder(tf.float32, shape=(None,len(LabelMask)), name="labels")
    traineddata = [] 
    if data is not None:
        for i, item in enumerate(data):
            item = np.array(item)
            traineddata.append(tf.placeholder(tf.float32, shape=item.T.shape, name="transfer_data{0}".format(i)))
    inputs = {"images": images, "labels": labels}
    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn.model_fn('train', inputs, params, transferdata=traineddata)
    eval_model_spec = model_fn.model_fn('eval', inputs, params, reuse=True, transferdata=traineddata)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params,train_inputs, eval_inputs,inputs, args.restore_from, traineddata=traineddata, data=data)

