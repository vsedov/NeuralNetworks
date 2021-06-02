#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: backprop_inputs
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi


def main() -> None:

    dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    weights = np.array(
        [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
    )

    dinputs = np.dot(weights.T, dvalues)
    print("Dx of inputs, is equal to the weights, multiplied by previous inputs")
    print(dinputs)


if __name__ == "__main__":
    pi.install_traceback()
    main()
