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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np
import torch


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_features, self.out_features = in_features, out_features
        self.input_layer = input_layer
        self.params = {"weight": np.random.randn(out_features, in_features) * np.sqrt(2. / in_features),
                       "bias": np.zeros((out_features, 1))}
        self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros((out_features, 1))}
        self.x = np.zeros((in_features, 1))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        if self.input_layer:
            x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        self.x = x
        assert x.shape[1] == self.params["weight"].shape[1], 'Linear Module: input data dimension does not match'
        out = np.dot(x, self.params["weight"].T)
        out += self.params["bias"].T
        assert out.shape[1] == self.params["weight"].shape[0], 'Linear Module: output data dimension does not match'
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = np.dot(dout.T, self.x)
        self.grads['bias'] = np.sum(dout.T, axis=1, keepdims=True)
        dx = np.dot(dout, self.params['weight'])
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = np.zeros((self.out_features, self.in_features))
        self.grads['bias'] = np.zeros((self.out_features, 1))
        #######################
        # END OF YOUR CODE    #
        #######################


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = x * (x > 0)
        self.x = x
        assert x.shape == out.shape, "ReLU module"
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = (self.x > 0.)*1.
        dx = dx * dout
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x # (batch_size, n_classes)
        x_max = np.expand_dims(x.max(axis=1), axis=0).T # np.max(x, keepdims=True)
        exps = np.exp(x - x_max)
        sum_exps = np.expand_dims(np.sum(exps, axis=1) , axis=0).T
        out = exps / sum_exps
        self.softmax_out = out
        assert x.shape == out.shape, "Softmax module in & out shapes do not match"
        # assert out.sum(axis=1) == np.ones(out.shape[0])
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        N = self.softmax_out.shape[1]
        dx = self.softmax_out * (dout - np.matmul((dout * self.softmax_out), np.ones((N, N))))
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Handle RuntimeWarning: divide by zero encountered in log
        eps = np.zeros(y.shape) + 1e-10

        p = x[range(len(y)), y]
        p = np.maximum(eps, np.minimum(1 - eps, p))
        assert p.shape == y.shape, str(p.shape) + ":" + str(y.shape)
        out = -np.sum(np.log(p)) / y.shape[0]

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        eps = np.zeros(x.shape) + 1e-10
        x = np.maximum(eps, np.minimum(1 - eps, x))

        T = np.zeros(x.shape)
        T[np.arange(y.size), y] = 1
        dx = -(1/y.shape[0]) * (T / x)

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx