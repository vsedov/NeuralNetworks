#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-25 Viv Sedov
#
# File Name: softmaxNumpy.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"


import nnfs
import numpy as np
from frosch import hook
from pprintpp import pprint as pp

nnfs.init()


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons
        )  # This is just random generated results
        self.biases = np.zeros(
            (1, n_neurons)
        )  # If it is dying, but you can change that to a non zero

    def forward(self, inputs: list) -> np:
        self.outputs = np.dot(inputs, self.weights) + self.biases


class SoftmaxLayer:
    def forward(self, inputs) -> np:
        expValues = np.exp(inputs - np.maximum(inputs))
        expSum = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.outputs = expSum


def softmaxNumpyBatch() -> np:
    layerOutput = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

    expValues = np.exp(layerOutput - np.maximum(layerOutput, layerOutput))

    print(expValues)
    # These are normalised values


def softmaxNumpy() -> np:
    layerOutput = [4.8, 1.21, 2.385]

    expValues = np.exp(layerOutput)

    normValues = expValues / np.sum(expValues)
    pp([x := normValues, sum(x)])


def main() -> None:
    # softmaxNumpy()
    # print("\n")
    softmaxNumpyBatch()


if __name__ == "__main__":
    hook()
    main()
    """ You can show how numpy effects every value . but we can state hey how"""

    """Notes 
    when you have, some input, you want to prevent having something known as an overflow error . 
    Now this is rather annoying, and this is the main cause for code exploading or more so 
    that your gradient exploads
    """
