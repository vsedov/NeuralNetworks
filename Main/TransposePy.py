#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-23 Viv Sedov
#
# File Name: TransposePy.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook


def tester():
    inputs = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]
    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    bias = [2, 3, 0.5]

    print(np.dot(inputs, np.array(weights).T) + bias)

    # If you got rid fo the transpose for this laayer, you would end up getting an error .
    # Which is rather annoying if oyu ask me, but yeah, i guess thats how it is in theend


def main() -> None:

    tester()


if __name__ == "__main__":
    hook(theme="fruity")
    main()
