#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: Sparse_loss_2
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import nnfs
import numpy as np
import pyinspect as pi
from icecream import ic


def loss_calc(softmax: np.ndarray, class_target: np.ndarray) -> None:

    if len(class_target.shape) == 1:
        print("Shape of the class targets is 1", sep="\n")
        correct_confidences = softmax[(pointer := (range(len(softmax)))), class_target]

        for i in pointer:
            print(i)
    elif len(class_target.shape) == 2:
        print("Your shape for the class targets is 2", sep="\n")
        correct_confidences = np.sum(softmax * class_target, axis=1)

    print(correct_confidences, "\n")

    neg_loss = -np.log(correct_confidences)
    average_loss = np.mean(neg_loss)
    print("Your given loss is: ", average_loss, sep="\n")


def info_Ln_zero():
    ic(1e-7)
    ic(np.log(1e-7))
    ic(np.log(1 - 1e-7))
    ic(np.log(1))


def numpy_clip(values: np.ndarray) -> None:

    print("\n")
    ic("Before being Cliped: => ", values)
    print("\n")
    clipped_values = np.clip(values, 1e-2, 1 - 1e-3)
    ic("After cliped: => ", clipped_values)


def main() -> None:
    softmax_output = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.008]])
    class_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

    # When you have a 2d tensor
    loss_calc(softmax_output, class_targets)
    print("\n")

    # When you have a 1d Tensor
    loss_calc(softmax_output, np.array([0, 1, 1]))

    print("---\nLnInfo\n")
    info_Ln_zero()

    numpy_clip(np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]))


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main()
