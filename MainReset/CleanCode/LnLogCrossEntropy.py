#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-29 Viv Sedov
#
# File Name: LnLogCrossEntropy.py
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import math

import numpy as np
from icecream import ic
from pprintpp import pprint as pp


def sample_code() -> int:
    softmax_output = [0.7, 0.1, 0.2]
    target_output = [1, 0, 0]
    loss = -(
        math.log(
            softmax_output[0] * target_output[0]
            + softmax_output[1] * target_output[1]
            + softmax_output[2] * target_output[2]
        )
    )
    print(loss)
    pp([x := (math.log(softmax_output[0])), -x])

    np_loss()


def np_loss() -> None:
    b = 5.2
    print(b, x := (np.log(b)))
    print(np.exp(x))

    batch_data()


def batch_data() -> list:
    print("With batch data \n")
    softmax_output = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.008]])

    class_targets = [0, 1, 1]  # dog, cat , cat

    print(
        "Arg max is ", np.argmax(softmax_output, axis=1)
    )  # this is the max value through each index

    for targ_dix, distribution in zip(class_targets, softmax_output):
        print(distribution[targ_dix])
    # or you can do the following

    ic(range(len(softmax_output)), len(softmax_output))
    ic(softmax_output[range(len(softmax_output)), class_targets])

    batch_data_with_numpy()


def batch_data_with_numpy() -> np:
    print("\n")
    softmax_output = np.array(
        [
            [0.7, 0.1, 0.2],  # 1 0 0
            [0.1, 0.5, 0.4],  # 0 1 0
            [0.02, 0.9, 0.008],  # 0 1 0
        ]
    )
    x = range(len(softmax_output))
    class_target = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

    if len(class_target.shape) == 1:
        correct_confidence = softmax_output[x, class_target]
    elif len(class_target.shape) == 2:
        correct_confidence = np.sum(softmax_output * class_target, axis=1)
        print(correct_confidence)

    neg_log = -np.log(correct_confidence)
    average_loss = np.mean(neg_log)
    print(average_loss)


def main() -> None:
    batch_data()
    batch_data_with_numpy()


if __name__ == "__main__":
    main()
