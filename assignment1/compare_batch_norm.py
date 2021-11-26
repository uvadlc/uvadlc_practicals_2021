################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested
    hyperparameters = ["128", "256,128", "512,256,128"]
    results_dict = {}
    for hyperp in hyperparameters:
        results_dict[hyperp] = {}
        results_dict[hyperp][True] = {}
        results_dict[hyperp][False] = {}
    for hyperp in hyperparameters:
        for use_bn in [True, False]:
            hidden_dims = list(map(int, hyperp.split(",")))
            model, val_accuracies, test_accuracy, logging_dict = \
                train_mlp_pytorch.train(hidden_dims, lr=0.1, use_batch_norm=use_bn, batch_size=128, epochs=20, seed=42,
                                        data_dir='data/')
            results_dict[hyperp][use_bn] = logging_dict

    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with open(results_filename, ) as f:
        results_dict = json.load(f)
    hyperparameters = ["128", "256,128", "512,256,128"]
    ''' plot training accuracies '''
    x = [e + 1 for e in range(len(results_dict["128"]['true']['train_accuracies']))]
    sizes = (10, 5)
    plt.rcParams["figure.figsize"] = sizes
    plt.plot(x, results_dict["128"]['true']['train_accuracies'], '-b', label='BatchNorm=True: hidden_dims=[128]')
    plt.plot(x, results_dict["128"]['false']['train_accuracies'], '--b', label='BatchNorm=False: hidden_dims=[128]')

    plt.plot(x, results_dict["256,128"]['true']['train_accuracies'], '-r', label='BatchNorm=True: hidden_dims=[256,128]')
    plt.plot(x, results_dict["256,128"]['false']['train_accuracies'], '--r', label='BatchNorm=False: hidden_dims=[256,128]')

    plt.plot(x, results_dict["512,256,128"]['true']['train_accuracies'], '-g',
             label='BatchNorm=True: hidden_dims=[512,256,128]')
    plt.plot(x, results_dict["512,256,128"]['false']['train_accuracies'], '--g',
             label='BatchNorm=False: hidden_dims=[512,256,128]')

    plt.grid(axis='x', color='0.95')
    plt.legend()

    plt.title('Train accuracies')
    plt.savefig('./train_accuracies_batchnorm.png')
    plt.clf()

    ''' plot validation accuracies '''
    plt.plot(x, results_dict["128"]['true']['val_accuracies'], '-b', label='BatchNorm=True: hidden_dims=[128]')
    plt.plot(x, results_dict["128"]['false']['train_accuracies'], '--b', label='BatchNorm=False: hidden_dims=[128]')

    plt.plot(x, results_dict["256,128"]['true']['val_accuracies'], '-r', label='BatchNorm=True: hidden_dims=[256,128]')
    plt.plot(x, results_dict["256,128"]['false']['val_accuracies'], '--r',
             label='BatchNorm=False: hidden_dims=[256,128]')

    plt.plot(x, results_dict["512,256,128"]['true']['val_accuracies'], '-g',
             label='BatchNorm=True: hidden_dims=[512,256,128]')
    plt.plot(x, results_dict["512,256,128"]['false'
    ]['val_accuracies'], '--g',
             label='BatchNorm=False: hidden_dims=[512,256,128]')

    plt.grid(axis='x', color='0.95')
    plt.legend()

    plt.title('Validation accuracies')
    plt.savefig('./validation_accuracies_batchnorm.png')

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.json'  # 'results.txt'
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)