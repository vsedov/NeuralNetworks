#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-31 Viv Sedov
#
# File Name: AccuracyCalc.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook
from pprintpp import pprint as pp


def accuracy():

    # this input data was all over the place and was very offputing when i had a look at it
    softmax_output = np.array([[0.7, 0.2, 0.1], [0.5, 0.1, 0.4], [0.02, 0.9, 0.08]])
    class_target = np.array([0, 1, 1])

    predictions = np.argmax(softmax_output, axis=1)
    # Index, im pretty certain .

    print("Prediction")
    pp(predictions)

    print(f"{class_target.shape} given shape")

    if len(class_target.shape) == 2:
        class_target = np.argmax(class_target, axis=1)
    print("Class targets, after arg max ")
    pp(class_target)

    print(f"{predictions} == {class_target}")

    outputs = predictions == class_target
    pp(outputs)
    acc = np.mean(outputs)
    print(acc)


def main() -> None:
    accuracy()


if __name__ == "__main__":
    hook()
    main()
