#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: acc_calc
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import nnfs
import numpy as np
import pyinspect as pi

nnfs.init()


def main() -> None:
    softmax_output = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.008]])
    class_targets = np.array([0, 0, 1])

    # Index in which the most max value is

    prediction = np.argmax(softmax_output, axis=1)
    print("ArgMax Value is :", prediction)
    print("Target ReqValue is :", class_targets)
    # With this we look at how off this infomation is

    if len(class_targets.shape) == 2:
        class_targets = np.argmax(class_targets, axis=1)

    print("Mean of this infomation ", prediction == class_targets)
    accuracy = np.mean(prediction == class_targets)

    print("Total loss is ", accuracy)


if __name__ == "__main__":
    pi.install_traceback()
    main()
