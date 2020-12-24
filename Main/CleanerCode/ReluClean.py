#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-24 Viv Sedov
#
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import nnfs
import numpy as np
from frosch import hook
from nnfs.datasets import spiral_data
from pprintpp import pprint as pp


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


# Very simple, but very nice to do as well .
class ActivationReLU:
    def forward(self, inputs) -> np:
        self.outputs = np.maximum(0, inputs)


def main() -> np:

    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2]

    X, y = spiral_data(samples=100, classes=3)  # This is your data

    layer1 = DenseLayer(2, 5)
    activation1 = ActivationReLU()
    layer1.forward(X)
    activation1.forward(layer1.outputs)

    pp(activation1.outputs[:5])  # When optimiser does things, ti just does it for you .
    print("\n")
    pp(activation1.outputs)  # When optimiser does things, ti just does it for you .


if __name__ == "__main__":
    hook()

    nnfs.init()  # This does a defualt data type, for us, which is rather nice .
    main()
