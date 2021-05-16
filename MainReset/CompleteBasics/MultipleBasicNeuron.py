#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: MultipleBasicNeuron
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import pyinspect as pi
from icecream import ic

import numpy as np


def version1():
    inputs = [1, 2, 3, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    bias = [2, 3, 0.5]

    print(np.dot(weights, inputs) + bias)


def main():

    inputs = [1, 2, 3, 2.5]

    weights1 = [0.2, 0.8, -0.5, 1]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]

    bias1, bias2, bias3, = (
        2,
        3,
        0.5,
    )

    # We have to push forward infomation when you have some bias
    outputs = [
        # Neuron1
        inputs[0] * weights1[0]
        + inputs[1] * weights1[1]
        + inputs[2] * weights1[2]
        + inputs[3] * weights1[3]
        + bias1,
        # Neuron2
        inputs[0] * weights2[0],
        inputs[1] * weights2[1],
        inputs[2] * weights2[2],
        inputs[3] * weights2[3] + bias2,
        # Neuron3
        inputs[0] * weights3[0]
        + inputs[1] * weights3[1]
        + inputs[2] * weights3[2]
        + inputs[3] * weights3[3]
        + bias3,
    ]

    ic(outputs)

    version1()


if __name__ == "__main__":
    pi.install_traceback()
    main()
