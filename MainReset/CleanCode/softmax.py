#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-25 Viv Sedov
#
# File Name: softmax.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import math
import random


def pointer() -> list:
    layerOutput = [4.8, 1.21, 2.385]
    E = math.e

    expValues = [pow(E, x) for x in layerOutput]

    normBase = sum(expValues)

    normValues = [value / normBase for value in expValues]
    print("\n")
    print(normValues)

    print("\n")
    print(sum(normValues))

    # This code is the same as belowo, just wanted to see how it is .


def softmax() -> list:
    layerOutput = [4.8, 1.21, 2.385]
    E = math.e

    expValues = [pow(E, x) for x in layerOutput]

    sumOfexpVal = sum(expValues)

    singularVal = [x / sumOfexpVal for x in expValues]

    print(
        singularVal
    )  # this is all of the out puts whta have been converted with eulers number.
    print(sum(singularVal))


def main() -> None:
    softmax()

    layer_output = [random.randint(1, 10) for _ in range(10)]

    point = [pow(math.e, x) for x in layer_output]
    final_val = ([x / sum(point) for x in point])
    print(final_val, sum(final_val))


if __name__ == "__main__":
    main()
