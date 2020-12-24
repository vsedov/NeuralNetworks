#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-24 Viv Sedov
#
# File Name: Relu.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import nnfs
import numpy as np
from frosch import hook

nnfs.init()


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons
        )  # This is just random generated results
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: list) -> np:
        self.outputs = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs) -> np:
        self.outputs = np.maximum(0, inputs)


def manualParse(X: list) -> DenseLayer:

    layer1 = DenseLayer(4, 5)
    layer2 = DenseLayer(5, 2)

    layer1.forward(X)

    layer2.forward(layer1.outputs)


def simpleRelu(X: list, inputs: list) -> list:

    output = []
    # This code is not good
    #    for i in inputs:
    #        if i > 0:
    #            output.append(i)
    #        elif i <= 0:
    #            output.append(0)
    #    pp(output)

    for i in inputs:
        output.append(max(0, i))
    print(output)  # Which is teh short version of your code


def main() -> np:

    X = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2]

    simpleRelu(X, inputs)
    # manualParse(X)


if __name__ == "__main__":
    hook(theme="fruity")
    main()
