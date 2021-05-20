#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: Loss_1
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import math

import nnfs
import numpy as np
import pyinspect as pi
from pprintpp import pprint as pp


def simple_exaple() -> None:

    softmax_output = [0.7, 0.1, 0.2]
    target_output = [1, 0, 0]

    # -----------------------------
    # Doing this with list comp and using the formulation for everything
    # $\sum(y_i_k \cdot \log(Y_i_k))$ log = ln()
    # -∑ (yₖ∙log(Yₖ))
    cross_entropy = [
        math.log(softmax_output[i]) * target_output[i]
        for i in range(len(softmax_output))
    ]
    print(-sum(cross_entropy))

    # ----------------
    # Notice how this is the same as the following formula
    # $-log(Y_k_i) $

    loss = -(math.log(softmax_output[0]) * target_output[0])
    print(loss)


def batch_example_wo_numpy(softmax_output: list) -> None:
    class_targets = [0, 1, 1]

    pp(softmax_output)
    pp(class_targets)
    # In this case we are mapping the inex for the given infomation
    for targ_index, distribution in zip(class_targets, softmax_output):
        print(distribution[targ_index])


def batch_simplified(softmax_output: list, class_target: list) -> list:
    # Numpy indexing with row space and row infomation
    pp(softmax_output[[0, 1, 2], class_target])

    pp(softmax_output)
    print(range(len(softmax_output)))
    # A better way of doing this would be using range in this case

    print("\n: Better version of doing this part_1")

    print(softmax_output[range(len(softmax_output)), class_target])


def arg_max_stuff(softmax: list, target: list) -> None:
    print("Targets are ", target, sep="\n")
    print("Argmax is :", np.argmax(softmax, axis=1), sep="\n")
    print("The argmax for this is the same as the class target")


def main() -> None:
    # simple_exaple()
    # ==============================
    # class_target is the index at this given point
    class_targets = [0, 1, 1]
    #                           1    0    0      0     1    0     0      1     0
    softmax_output = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.008]])
    # batch_example_wo_numpy(softmax_output)

    batch_simplified(softmax_output, class_targets)
    print("\n")
    arg_max_stuff(softmax_output, class_targets)


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main()
