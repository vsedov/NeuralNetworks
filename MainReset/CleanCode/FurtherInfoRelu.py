#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-24 Viv Sedov
#
# File Name: FurtherInfoRelu.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from matplotlib import pyplot


def plotter() -> pyplot:
    pyplot.style.use("ggplot")
    inputs = list(range(-19, 19))
    outputs = list(map(lambda x: max(0.01 * x, x),
                       inputs))  # and this becomes leaky relu
    pyplot.plot(inputs, outputs)
    pyplot.show()


def plotter_version_2() -> pyplot:
    pyplot.style.use("ggplot")
    inputs = list(range(-19, 19))
    outputs = list(map(lambda x: 1 + np.exp(-x), inputs))
    pyplot.plot(inputs, outputs)
    pyplot.show()


def main() -> None:
    plotter_version_2()


if __name__ == "__main__":
    main()
