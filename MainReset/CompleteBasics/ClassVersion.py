#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: ClassVersion
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import pyinspect as pi
import nnfs
from nnfs.datasets import spiral_data
from pprintpp import pprint as pp

import numpy as np


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        # weights are made for you - transpose not needed
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.bias


class ActivationRelu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


def main() -> None:

    # y woudl be the classification that you are trying to get
    X, y = spiral_data(samples=100, classes=3)

    # pp(X[:5])

    dense1 = LayerDense(2, 3)
    activation1 = ActivationRelu()

    dense1.forward(X)
    activation1.forward(dense1.output)
    pp(activation1.output[:5])


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main()
