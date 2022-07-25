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
from icecream import ic
from nnfs.datasets import spiral_data
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


class Activation_ReLU:
    def forward(self, inputs) -> None:
        self.output = np.maximum(0, inputs)


class SoftmaxLayer:
    def forward(self, inputs) -> np:
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = expValues / np.sum(expValues, axis=1, keepdims=True)


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
    # softmaxNumpyBatch()
    #
    x, y = spiral_data(samples=100, classes=3)
    dense_1 = DenseLayer(2, 3)
    activation_1 = Activation_ReLU()

    dense_2 = DenseLayer(3, 3)

    activation_2 = SoftmaxLayer()

    dense_1.forward(x)
    activation_1.forward(dense_1.outputs)
    dense_2.forward(activation_1.output)
    activation_2.forward(dense_2.outputs)

    # you would have to make sure that the axis is 1 .
    ic(np.sum(activation_2.outputs[:5], axis=1))


if __name__ == "__main__":
    main()
    """ You can show how numpy effects every value . but we can state hey how"""
    """Notes
    when you have, some input, you want to prevent having something known as an overflow error .
    Now this is rather annoying, and this is the main cause for code exploading or more so
    that your gradient exploads
    """
