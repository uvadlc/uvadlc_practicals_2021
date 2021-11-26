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
# Date Created: 2021-11-11
###############################################################################
"""
This file contains the implementation of all necessary corruption functions for the experiments.
You don't have to change anything here, but make yourself familiar with the functions.
"""
import torch
import torchvision
from torchvision import transforms


def gaussian_noise_transform(severity=1):
    """
    Creates a Gaussian noise corruption function with the specified severity.

    Args:
        severity: Scale of the corruption to use, has to be between 1 and 5.
    """
    std = [0.04, 0.07, 0.10, 0.15, 0.25][severity - 1]
    return GaussianNoiseTransformation(std=std)


def gaussian_blur_transform(severity=1):
    """
    Creates a Gaussian blur corruption function with the specified severity.

    Args:
        severity: Scale of the corruption to use, has to be between 1 and 5.
    """
    sigma = [0.4, 0.6, 0.8, 1.1, 1.5][severity - 1]
    return transforms.GaussianBlur(kernel_size=11, sigma=sigma)


def contrast_transform(severity=1):
    """
    Creates a contrast reduction corruption function with the specified severity.

    Args:
        severity: Scale of the corruption to use, has to be between 1 and 5.
    """
    c = [0.75, 0.6, 0.5, 0.4, 0.3][severity - 1]
    return transforms.ColorJitter(contrast=(c,c))


def jpeg_transform(severity=1):
    """
    Creates a JPEG compression corruption function with the specified severity.

    Args:
        severity: Scale of the corruption to use, has to be between 1 and 5.
    """
    quality = [90, 70, 50, 30, 10][severity - 1]
    return JPEGTransformation(quality)


class GaussianNoiseTransformation(object):
    """
    Transformation class of applying Gaussian noise to an image.
    To be used within a torch transformation composition.
    """

    def __init__(self, std=1.):
        self.std = std
        
    def __call__(self, img):
        img = img + torch.randn_like(img) * self.std
        img = img.clamp_(min=0, max=1)
        return img


class JPEGTransformation(object):
    """
    Transformation class of en- and decoding images with JPEG.
    To be used within a torch transformation composition.
    """

    def __init__(self, quality=90):
        self.quality = quality

    def __call__(self, img):
        img = (img * 255).to(torch.uint8)
        img = torchvision.io.encode_jpeg(img, quality=self.quality)
        img = torchvision.io.decode_jpeg(img)
        img = img.float() / 255.0
        return img