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
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # This is just random generated results
        self.biases = np.zeros((1, n_neurons))
        # If it is dying, but you can change that to a non zero

    def forward(self, inputs: list) -> np:
        self.outputs = np.dot(inputs, self.weights) + self.biases


# Very simple, but very nice to do as well .
class ActivationReLU:
    def forward(self, inputs) -> np:
        self.outputs = np.maximum(0, inputs)


class ActivationSoftMax:
    def forward(self, inputs) -> np:
        expVal = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Cool this works - Take Value away and keeep that dimension .
        # This prevents overflow error

        probability = expVal / np.sum(expVal, axis=1, keepdims=True)

        self.outputs = probability


def main() -> None:

    X, y = spiral_data(samples=100, classes=3)  # This is your data

    dense1 = DenseLayer(2, 3)
    activation1 = ActivationReLU()
    dense2 = DenseLayer(3, 3)
    # The output layer is what ever or how ever many classes that you have

    activation2 = ActivationSoftMax()

    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)

    pp([x := activation2.outputs[:5], type(x)])


if __name__ == "__main__":
    hook()

    nnfs.init()  # This does a defualt data type, for us, which is rather nice .
    main()
