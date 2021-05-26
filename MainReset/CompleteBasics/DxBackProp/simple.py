#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: simple Diriv Basics, in Code form
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import matplotlib.pyplot as plt
import pyinspect as pi


def f(x: int) -> int:
    # f(x) = 2x²
    return 2 * x ** 2


def parser() -> None:
    p2_delta = 1e-3
    x1 = 1
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    app_diriv = (y2 - y1) / (x2 - x1)
    print(app_diriv)


def simple_understanding() -> None:
    for i in range(5):
        p2_delta = 1e-05
        x1 = i
        x2 = x1 + p2_delta

        y1 = f(x1)

        y2 = f(x2)

        diriv = (y2 - y1) / (x2 - x1)
        b = y2 - diriv * x2

        def approximate_tangent_line(x: int, diriv: int) -> None:
            return (diriv * x) + y2 - diriv * x2

        print(f" x = {x1} , y =  {y1}, {diriv}")
        print("\n")
        print(f"{x1} {diriv} Bth value {b} -> ", approximate_tangent_line(i, diriv))

        to_plot = [x1 - 0.9, x1, x1 + 0.9]
        plt.plot(
            [point for point in to_plot],
            [approximate_tangent_line(point, diriv) for point in to_plot],
        )
    plt.show()


def main() -> None:
    print([f(i) for i in range(0, 5)])

    # Showing how diriv works, with teh basics
    parser()

    print("-------------------------------- Simple Understanding \n")

    simple_understanding()


if __name__ == "__main__":
    pi.install_traceback()
    main()
