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
from icecream import ic
from nnfs.datasets import spiral_data

nnfs.init()


class DenseLayer:

    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons)  # This is just random generated results
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: list) -> np:
        self.outputs = np.dot(inputs, self.weights) + self.biases


class ActivationRelu:

    def forward(self, inputs: list) -> list:
        self.output = np.maximum(0, inputs)


def main():
    x, y = spiral_data(100, 3)  # 100 feature sets of 3 class sses():
    layer_1 = DenseLayer(2, 3)
    activation_1 = ActivationRelu()
    layer_1.forward(x)
    ic(layer_1.outputs)
    activation_1.forward(layer_1.outputs)
    ic(activation_1.output)


if __name__ == "__main__":
    main()
