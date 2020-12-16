#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-12-16 Viv Sedov
#
# File Name: Layers_of_Neuron.py
# Distributed under terms of the MIT license.
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import better_exceptions
import numpy as np
from frosch import hook
from pprintpp import pprint as pp

better_exceptions.hook()
hook()


def threeNeurons() -> list:
    inputs = [1, 2, 3, 2.5]

    weights1 = [0.2, 0.8, -0.5, 1]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]

    biases1 = 2
    biases2 = 3
    biases3 = 0.5
    npParser(inputs, weights1, weights2, weights3, biases1, biases2, biases3)

    output = [
        # Neuron 1
        inputs[0] * weights1[0]
        + inputs[1] * weights1[1]
        + inputs[2] * weights1[2]
        + inputs[3] * weights1[3]
        + biases1,
        # Neuron 2
        inputs[0] * weights2[0]
        + inputs[1] * weights2[1]
        + inputs[2] * weights2[2]
        + inputs[3] * weights2[3]
        + biases2,
        # Neuron 3
        inputs[0] * weights3[0]
        + inputs[1] * weights3[1]
        + inputs[2] * weights3[2]
        + inputs[3] * weights3[3]
        + biases3,
    ]
    print(output)


def npParser(input, weights1, weights2, weights3, biases1, biases2, biases3):

    dotVector1 = np.dot(input, weights1) + biases1
    dotVector2 = np.dot(input, weights2) + biases2
    dotVector3 = np.dot(input, weights3) + biases3

    pp([dotVector1, dotVector2, dotVector3])


def main() -> None:
    threeNeurons()


if __name__ == "__main__":
    main()
