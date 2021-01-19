#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2021-01-18 Viv Sedov
#
# File Name: WholeProcess.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook
from pprintpp import pprint as pp


def main() -> None:

    dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    inputs = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])

    weights = np.array(
        [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
    ).T

    # Shape is 1, Neurons , so we have this empty array on teh side
    biases = np.array([[2, 3, 0.5]])

    layer_outputs = np.dot(inputs, weights) + biases
    relu_outputs = np.maximum(0, layer_outputs)

    drelu = relu_outputs.copy()
    drelu[layer_outputs <= 0] = 0
    pp(drelu)

    # We will have to fix the shape for this
    dinputs = np.dot(drelu, weights.T)
    dweights = np.dot(inputs.T, drelu)

    dbiases = np.sum(drelu, axis=0, keepdims=True)

    weights += -0.001 * dweights
    biases += -0.001 * dbiases
    print("\n")
    pp(["Updated Param", weights, biases])


if __name__ == "__main__":
    hook()
    main()
