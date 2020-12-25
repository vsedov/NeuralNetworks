#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-24 Viv Sedov
#
# File Name: FurtherInfoRelu.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

from frosch import hook
from matplotlib import pyplot


def plotter() -> pyplot:
    pyplot.style.use("ggplot")
    inputs = [x for x in range(-19, 19)]
    outputs = list(
        map(lambda x: max(0.01 * x, x), inputs)
    )  # and this becomes leaky relu
    pyplot.plot(inputs, outputs)
    pyplot.show()


def main() -> None:
    plotter()


if __name__ == "__main__":
    hook()
    main()
