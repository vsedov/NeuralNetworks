#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-19 Viv Sedov
#
# File Name: Vizulizer.py
from __future__ import absolute_import

__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import better_exceptions
import numpy as np
from frosch import hook
from pprintpp import pprint as pp

better_exceptions.hook()
hook()

import matplotlib.pyplot as plt


def dotProduct() -> np:
    plt.plot([1, 1])
    plt.ylabel("Some infomation")

    plt.show()


def nump() -> np:
    a = [1, 2, 3, 4]

    print(np.expand_dims(np.array(a), axis=0))

    a = [1, 2, 3]
    b = [2, 3, 4]

    a = np.array([a])
    b = np.array([b]).T
    print(np.dot(a, b))


def main() -> None:
    # dotProduct()
    nump()


if __name__ == "__main__":
    main()
