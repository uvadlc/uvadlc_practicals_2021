###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Adapted: 2021-11-11
###############################################################################


import os
import numpy as np
import torch
import torch.utils.data as data


class TextDataset(data.Dataset):

    def __init__(self, filename, seq_length, step_size=10, random_select=True):
        assert os.path.splitext(filename)[1] == ".txt"
        self._seq_length = seq_length
        self._step_size = step_size
        self._random_select = random_select
        self._data = open(filename, 'r').read()
        self._chars = sorted(list(set(self._data)))
        self._data_size, self._vocabulary_size = len(self._data), len(self._chars)
        print(f"Initialize dataset with {self._data_size} characters, {self._vocabulary_size} unique.")
        self._char_to_ix = {ch: i for i, ch in enumerate(self._chars)}
        self._ix_to_char = {i: ch for i, ch in enumerate(self._chars)}
        self._offset = 0

    def __getitem__(self, item):
        idx = item * self._step_size
        if self._random_select:
            idx += np.random.randint(self._step_size)
        sentence = [self._char_to_ix[ch] for ch in self._data[idx:idx + self._seq_length + 1]]
        sentence = torch.LongTensor(sentence)
        return sentence

    def convert_to_string(self, char_ix):
        return ''.join(self._ix_to_char[ix] for ix in char_ix)

    def __len__(self):
        return (len(self._data) - self._seq_length - 1) // self._step_size - 1

    @property
    def vocabulary_size(self):
        return self._vocabulary_size


def text_collate_fn(batch):
    inputs = torch.stack([item[:-1] for item in batch], dim=1)
    targets = torch.stack([item[1:] for item in batch], dim=1)
    return inputs, targets