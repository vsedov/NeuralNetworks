#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: BackProp2
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi
from pprintpp import pprint as pp


def main() -> None:
    # For this purpose of this example
    # Vector of ones will be pased through
    dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    # So this would be the base layer, or what ever inofmation we have

    # We have 4 inputs, and because of that we would have 4 weights
    weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]])

    print("\n")
    weights = weights.T

    pp(weights)
    """
    sum weights related to the given input
    * gradient related to the given neuron
    """
    print("\n")

    dx0 = sum([
        weights[0][0] * dvalues[0][0],
        weights[0][1] * dvalues[0][1],
        weights[0][2] * dvalues[0][2],
    ])
    dx1 = sum([
        weights[1][0] * dvalues[0][0],
        weights[1][1] * dvalues[0][1],
        weights[1][2] * dvalues[0][2],
    ])

    dx2 = sum([
        weights[2][0] * dvalues[0][0],
        weights[2][1] * dvalues[0][1],
        weights[2][2] * dvalues[0][2],
    ])

    dx3 = sum([
        weights[3][0] * dvalues[0][0],
        weights[3][1] * dvalues[0][1],
        weights[3][2] * dvalues[0][2],
    ])
    # This is pretty much doing what this had done before
    print("\n")
    dinputs = np.array([dx0, dx1, dx2, dx3])

    print("Version 1 ", dinputs)

    # This is the thing that you had learnt about the transpose
    # Of the given value
    doter = np.dot(dvalues, weights.T)
    pp(doter)


if __name__ == "__main__":
    pi.install_traceback()
    main()
