#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: BatchInfomation
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

# import pyinspect as pi

import numpy as np
from icecream import ic


def example_input_data():
    inputs1 = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]
    weights1 = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    output1 = np.dot(np.array(inputs1), np.array(weights1).T) + [2, 3, 0.5]
    ic(output1)
    # you would also dont want to be repeating infomation

    # This is the batch of outputs
    # you have to transpose this infomation .
    #  when you ahve a batch as the shapes no longer match.

    # You want everthing to match what you have at the given time .

    # Another layer , size.


def main():
    example_input_data()


if __name__ == "__main__":
    main()
