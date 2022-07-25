#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: BackProp_relu
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi
from icecream import ic


def version_1() -> None:

    # Output information from the layer
    z = np.array([[1, 2, -3, -4], [2, -7, -1, 2], [-1, 2, 4, -1]])

    dvalues = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # We do relu derivative
    drelu = np.zeros_like(z)
    # If greater than zero, let it equal to one , else be zero
    drelu[z > 0] = 1
    ic(drelu, z)

    print("Dirivative of relu ", drelu, sep="\n")
    print("\n")
    # Do the chain rule
    drelu *= dvalues
    print("Drelu * dvalues ", drelu, sep="\n")


def version_2() -> None:
    z = np.array([[1, 2, -3, -4], [2, -7, -1, 2], [-1, 2, 4, -1]])

    dvalues = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # relu activation function derivative
    # with the chain rule applied

    drelu = dvalues.copy()
    drelu[z <= 0] = 0
    print(drelu)


def main() -> None:

    # version_1()
    # print("\n")
    version_1()


if __name__ == "__main__":
    pi.install_traceback()
    main()
