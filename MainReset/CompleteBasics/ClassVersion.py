#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: ClassVersion
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import nnfs
import numpy as np
import pyinspect as pi
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        # weights are made for you - transpose not needed
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.bias


class ActivationRelu:
    def forward(self, inputs: np.ndarray):
        self.output = np.maximum(0, inputs)


class ActivationSoftMax:
    def forward(self, inputs: np.ndarray):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


def main() -> None:

    # y woudl be the classification that you are trying to get
    X, y = spiral_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationRelu()

    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftMax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(
        "Without Checking if they added up to one softmax: \n", activation2.output[:5]
    )
    print(
        "\n, Checking that they would all added up to one: \n",
        np.sum(activation2.output[:5], axis=1, keepdims=True),
    )


if __name__ == "__main__":
    pi.install_traceback()
    main()
