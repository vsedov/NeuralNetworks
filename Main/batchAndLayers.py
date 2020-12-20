#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-20 Viv Sedov
#
# File Name: batchAndLayers.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook
from pprintpp import pprint as pp


def calCu() -> np:  # Shape Error

    inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, 0.8, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    bias = [2, 3, 0.5]  # We are not changing anything else here

    # Index of one - dot needs to match Index Zero
    pp(np.dot(weights, inputs) + bias)


def calCuFixShape() -> np:  # Shape Error

    inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, 0.8, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    bias = [2, 3, 0.5]  # We are not changing anything else here

    # Index of one - dot needs to match Index Zero

    # Convert weights to numpy array

    # So this is what you would do

    output = np.dot(inputs, np.array(weights).T) + bias

    pp(output)


def main() -> None:
    # calCu()
    # print("\n")

    calCuFixShape()


if __name__ == "__main__":
    hook(theme="fruity")
    main()
