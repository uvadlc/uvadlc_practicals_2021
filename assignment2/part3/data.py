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
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

from warnings import warn

import torch
from torch_geometric.data.batch import Batch
from torch_geometric.data.dataset import Dataset
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_adj

# Constants for the QM9 problem
MAX_NUM_NODES = 29  # The larget molecule has 29 atoms
Z_ONE_HOT_DIM = 5  # The one hot encoding of the element (z) has 5 unique values
EDGE_ATTR_DIM = 4  # The one hot encoding of the edges have 4 unique values
LABEL_INDEX = 7  # The energy of atomization at 0K exists at index 7
FLAT_INPUT_DIM = 3509  # The largest molecule in QM9 with all the node features and edges flattened is this long


def get_node_features(molecules: Batch) -> torch.Tensor:
    """return the node features permitted for the problem. the features are one hot encodings of the atomic number.

    Args:
        molecules: pytorch geometric batch of molecules

    Returns:
        z: a one hot tensor based on the node's atomic number. (batch, Z_ONE_HOT_DIM)
    """
    return molecules.x[:, :Z_ONE_HOT_DIM]


def get_labels(molecules: Batch) -> torch.Tensor:
    """return the labels for our problem. the labels are u0.

    Args:
        molecules: pytorch geometric batch of molecules

    Returns:
        u0 labels: a tensor of labels (batch, 1)
    """
    return molecules.y[:, LABEL_INDEX]


def get_qm9(data_dir: str, device="cpu") -> tuple[Dataset, Dataset, Dataset]:
    """Download the QM9 dataset from pytorch geometric. Put it onto the device. Split it up into train / validation / test.

    Args:
        data_dir: the directory to store the data.
        device: put the data onto this device.

    Returns:
        train dataset, validation dataset, test dataset.
    """
    dataset = QM9(data_dir)

    # Permute the dataset
    try:
        permu = torch.load("permute.pt")
        dataset = dataset[permu]
    except FileNotFoundError:
        warn("Using non-standard permutation since permute.pt does not exist.")
        dataset, _ = dataset.shuffle(return_perm=True)

    # z score / standard score targets to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, LABEL_INDEX].item(), std[:, LABEL_INDEX].item()

    # Move the data to the device (it should fit on lisa gpus)
    dataset.data = dataset.data.to(device)

    len_train = 100_000
    len_val = 10_000

    train = dataset[:len_train]
    valid = dataset[len_train : len_train + len_val]
    test = dataset[len_train + len_val :]

    assert len(dataset) == len(train) + len(valid) + len(test)

    return train, valid, test


def get_z_one_hot(molecules: Batch) -> torch.Tensor:
    """accesses the node features from batch of molecules and converts them into a dense, padded tensor.

    Args:
        molecules: a batch of molecules from pytorch geometric

    Returns:
        dense one hot vector representing atomic type
    """
    batch_size = molecules.batch.unique().numel()
    one_hots = get_node_features(molecules)
    _, counts = molecules.batch.unique(return_counts=True)
    row_position = torch.cat(
        [torch.arange(c, device=molecules.batch.device) for c in counts]
    )
    indices = torch.stack([molecules.batch, row_position])
    z_one_hot = torch.sparse_coo_tensor(
        indices=indices,
        values=one_hots,
        size=(batch_size, MAX_NUM_NODES, Z_ONE_HOT_DIM),
    ).to_dense()
    return z_one_hot


def get_dense_adj(molecules: Batch) -> torch.Tensor:
    """accesses the edge index and attr from a batch of molecules and converts them into a dense, padded adjacency matrix.

    Args:
        molecules: pytorch geometric batch

    Returns:
        dense, padded adjacency matrix with edge attr
    """
    dense_adj = to_dense_adj(
        molecules.edge_index, molecules.batch, molecules.edge_attr, MAX_NUM_NODES
    )
    return dense_adj


def get_mlp_features(molecules: Batch) -> torch.Tensor:
    """accesses the batch and produces a padded, flattened tensor suitable for an mlp.

    Args:
        molecules: pytorch geometric batch

    Returns:
        dense, padded, flattened input features
    """
    z_one_hot = get_z_one_hot(molecules)
    dense_adj = get_dense_adj(molecules)
    x = torch.cat([z_one_hot.flatten(1), dense_adj.flatten(1)], dim=1)
    return x
