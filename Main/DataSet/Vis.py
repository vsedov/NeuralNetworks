#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-22 Viv Sedov
#
# File Name: Vis.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

from frosch import hook
def ploter() -> plt:

    X, y = spiral_data(samples=100, classes=3)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    plot(X, y)


def plotter2() ->plt:

    X, y = spiral_data(samples=100, classes=3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brig")
    plt.shotw()


def main() -> None:

    plotter2()



if __name__ == "__main__":
  hook(theme="fruity")
  main()
