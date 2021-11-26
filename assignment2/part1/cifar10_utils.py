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
This module implements utility functions for downloading and reading CIFAR10 data.
You don't need to change anything here, but make yourself familiar with the functions.
"""
import torch

# tools used or loading cifar10 dataset
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms


def get_train_validation_set(data_dir, validation_size=5000):
    """
    Returns the training and validation set of CIFAR10.

    Args:
        data_dir: Directory where the data should be stored.
        validation_size: Size of the validation size

    Returns:
        train_dataset: Training dataset of CIFAR10
        val_dataset: Validation dataset of CIFAR10
    """

    mean = (0.491, 0.482, 0.447)
    std  = (0.247, 0.243, 0.262)

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    # We need to load the dataset twice because we want to use them with different transformations
    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=val_transform)
    
    # Subsample the validation set from the train set
    if not 0 <= validation_size <= len(train_dataset):
        raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
            len(train_dataset), validation_size))
        
    train_dataset, _ = random_split(train_dataset, 
                                    lengths=[len(train_dataset) - validation_size, validation_size],
                                    generator=torch.Generator().manual_seed(42))
    _, val_dataset = random_split(val_dataset, 
                                  lengths=[len(val_dataset) - validation_size, validation_size],
                                  generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset


def get_test_set(data_dir, augmentation=None):
    """
    Returns the test dataset of CIFAR10 with optional additional transformation/augmentation.

    Args:
        data_dir: Directory where the data should be stored
        augmentation [optional]: A torchvision transformation that should be added to the 
                                 transformation set (e.g. one of the corruption functions)
    Returns:
        test_dataset: The test dataset of CIFAR10.
    """

    mean = (0.491, 0.482, 0.447)
    std  = (0.247, 0.243, 0.262)

    test_transform = [transforms.ToTensor()]
    if augmentation is not None:
        test_transform.append(augmentation)
    test_transform.append(transforms.Normalize(mean, std))
    test_transform = transforms.Compose(test_transform)

    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return test_dataset