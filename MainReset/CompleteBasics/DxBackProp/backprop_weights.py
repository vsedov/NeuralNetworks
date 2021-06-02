#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: backprop_weights
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi


def main() -> None:

    dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    inputs = np.array([[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]])

    # you have3 sets of input and they are 4 samples for each one  aka
    # 4 features with each set

    dweights = np.dot(inputs.T, dvalues)
    # Multiply that infomation from the side pointer
    print("Partial Dx : ", dweights)


if __name__ == "__main__":
    pi.install_traceback()
    main()
