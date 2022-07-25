#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: FullPassExample
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
from icecream import ic  # pyright:ignore


def main() -> None:
    """Full pass example"""

    # passed in gradient from the next layer
    # array of incremental gradient for this example used, but this will not be
    # the case later on
    dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    # We have 3 sets of inputs and 4 sample features

    inputs = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])

    # Recall we would have to make sure that the shape matches
    weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]]).T

    print("Original Weights ", weights)
    # we have one bias for each Neuron

    bias = np.array([[2, 3, 0.5]])
    print("Original Biases ", bias)

    # Now we would do a forward parse on each of these neruons

    layer_output = np.dot(inputs, weights) + bias

    print("Layer_output :", layer_output, sep="\n")
    print("\n")

    # recall that relu was max 0 else stays the same
    relu_outputs = np.maximum(0, layer_output)

    print("Relu outputs ", relu_outputs, sep="\n")

    # Back prop time
    # from the next layer down to the current layer when doing back prop

    drelu = relu_outputs.copy()
    # We do not want to change what the drelu outputs are

    drelu[layer_output <= 0] = 0

    # Dense layer

    # Derivative of inputs is weights
    dinputs = np.dot(drelu, weights.T)

    # Derivative of weights is inputs
    dweights = np.dot(inputs.T, drelu)

    # dbiases = sum values -> on teh frist axis, meaning that we know that it
    # will just be teh drelu, if you refer back to what the sumunation rule is

    dbiases = np.sum(drelu, axis=0, keepdims=True)

    print("\n")
    # output for partial dx of relu
    ic(drelu)

    print("\n")
    # output for dx for inputs
    ic(dinputs)

    print("\n")
    # outputs for dx for weights
    ic(dweights)

    print("\n")
    # outputs for dx for bias
    ic(dbiases)

    # Now we have to optimise the original weights and bias, with the new
    # infomation that we have

    weights += -0.001 * dweights
    bias += -0.001 * dbiases

    print("\nNew Infomation\n")
    ic(weights)
    ic(bias)


if __name__ == "__main__":
    main()
