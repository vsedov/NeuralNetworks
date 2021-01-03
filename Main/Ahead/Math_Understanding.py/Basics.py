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


def d(x):
    return 2 * x ** 2


def main() -> None:
    for i in range(5):
        p2_delta = 1e-05
        x1 = i
        x2 = x1 + p2_delta

        y1 = d(x1)
        y2 = d(x2)

        diriv = (y2 - y1) / (x2 - x1)
        b = y2 - diriv * x2

        """
        dy/dx -> diriv 
        y = mx + c 
        c = y - mx 
            
        tangent line is :
        
        mx* x1 + b  = y -> Tangehnt line
        """

        def approximate_tangent_line(x, diriv):
            return (diriv * x) + y2 - diriv * x2

        print(f" x = {x1} , y =  {y1}, {diriv}")
        print("\n")
        print(f"{x1} {diriv}", approximate_tangent_line(i, diriv))

        to_plot = [x1 - 0.9, x1, x1 + 0.9]
        plt.plot(
            [point for point in to_plot],
            [approximate_tangent_line(point, diriv) for point in to_plot],
        )
    plt.show()


if __name__ == "__main__":
    hook()
    main()