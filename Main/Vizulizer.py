#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-19 Viv Sedov
#
# File Name: Vizulizer.py
from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np


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


def test() -> np:
    inputs = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]
    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    bias = [2, 3, 0.5]

    inputs1 = np.array(inputs).T
    # This is the correct way of doing it
    print(np.dot(inputs, np.array(weights).T) + bias)

    # In this case you would have the inputs and weights first, or have that transposed i think this would be the better options


def main() -> None:
    # dotProduct()
    nump()
    test()


if __name__ == "__main__":
    main()
