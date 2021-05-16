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

import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
        # weights are made for you - transpose not needed
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        self.bias = np.zeros((n_inputs, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.bias


def main() -> None:

    X = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main()
