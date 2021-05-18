#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: SoftmaxInfomation
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import math

import nnfs
import numpy as np
import pyinspect as pi
from pprintpp import pprint as pp


def question(question: str):
    return pi.ask(question)


def example_version1(lst: list) -> None:
    exponate_list = [pow(math.e, i) for i in lst]
    print(exponate_list)


def with_numpy(layer_outputs: list) -> None:
    """
    Softmax with numpy

    Normalisation of list

    Parameters
    ----------
    layer_outputs : list
        1D list
    """
    exp_values = np.exp(layer_outputs)
    print("Exponentiated val: ", exp_values)

    # Normalize the values
    norm_values = exp_values / np.sum(exp_values)
    print("\nNormalized values : ", norm_values)

    print("\nSum of the normalized values : ", np.sum(norm_values))


def with_numpy_batch(layer_outputs: list) -> None:
    """
    batch softmax with numpy

    np array parse through convert to batch
    with softmax

    Parameters
    ----------
    layer_outputs : list
        np array or a 2d+ array
    """
    print("\n")
    pp(layer_outputs)
    print("Axis = None - does it indivudal -> ", np.sum(layer_outputs, axis=None), "\n")
    print("Axis = 0 - sum of columns -> ", np.sum(layer_outputs, axis=0), "\n")
    print("Axis = 1 - sum of rows -> ", np.sum(layer_outputs, axis=1), "\n")

    print(
        "Axis = 1 - sum of rows :: keepdims=True -> Retains shape of matrix \n",
        np.sum(layer_outputs, axis=1, keepdims=True),
    )

    print("\n-- Starting Softmax --\n")

    exp_values = np.exp(layer_outputs)

    print("Exponentiated val: \n", exp_values, "\n")

    exp_norm = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    print("Normalized value: \n", exp_norm[:5], "\n")

    print(
        "Sum of norm should equal to 1 -> \n",
        np.sum(exp_norm, axis=1, keepdims=True),
        "\n",
    )


def overflow_prevention(layer_output: list) -> None:
    """
    Overflow prevention - using V = U-Max(U)

    Overflow prevention that is occoured before list gets expon

    Parameters
    ----------
    layer_output : list
        np array list
    """

    exp_val_overflow_protection = np.exp(
        layer_output - np.max(layer_output, axis=1, keepdims=True)
    )
    print(
        "Exponentiated values with overflow protection: \n",
        exp_val_overflow_protection,
        "\n",
    )

    exp_normalised = exp_val_overflow_protection / np.sum(
        exp_val_overflow_protection, axis=1, keepdims=True
    )

    print("You notice that the values are the exact same as before:\n")
    pp(exp_normalised)


def exp_explosion(num: int = 1000):
    return np.exp(num)


def main() -> None:

    # When you have a singular list

    # ---------------------------------
    # layer_outputs = [4.8, 1.21, 2.385]
    # with_numpy(layer_outputs)
    # ---------------------------------

    # When you have a batch of infomation - axis become rather important

    # ---------------------------------
    layer_outputs_batch = np.array(
        [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]
    )
    with_numpy_batch(layer_outputs_batch)
    # ---------------------------------

    # Run time error , causing explosion .
    # ---------------------------------
    # exp_explosion()
    # ---------------------------------

    # Overflow prevention
    # ---------------------------------
    overflow_prevention(layer_outputs_batch)
    # ---------------------------------


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main()
