#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2021-01-01 Viv Sedov
#
# File Name: Basics.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import matplotlib.pyplot as plt
import numpy as np
from frosch import hook
from pprintpp import pprint as pp


def f(x):
    x2 = [pow(2 * x, 2) for x in x]
    container = []
    for varr in range(1, len(x)):
        container.append(x2[varr] - x2[varr - 1] / x[varr] - x[varr - 1])

    container.append(400)
    container = np.array(container)
    print(x, x2, container, sep="\n")

    print(x.shape, container.shape)

    plt.plot(x,container)
    plt.show()

    plt.plot(x,x2)
    plt.show()

def main() -> None:
    x = np.array(range(10))
    y = f(x)
    print(y)


if __name__ == "__main__":
    hook()
    main()
