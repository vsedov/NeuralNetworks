#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2021-01-02 Viv Sedov
#
# File Name: VectorClacStuff.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook
from pprintpp import pprint as pp


def forward():
    x = [1, -2, 3]
    w = [-3, -1, 2]

    bias = 1

    pointer = [x[i] * w[i] for i in range(len(x))]
    pointer_bias = sum(pointer) + bias
    # Interesting, i had not known that numpy doe sthis too

    relupointer = relu(pointer_bias)
    print(
        f"For values {pointer} we add a bias to given inputs to get {pointer_bias} and once put into relu, you would get {relupointer}"
    )


def relu(inputs):
    return max(inputs, 0.001)


def main() -> None:
    forward()


if __name__ == "__main__":
    hook()
    main()
