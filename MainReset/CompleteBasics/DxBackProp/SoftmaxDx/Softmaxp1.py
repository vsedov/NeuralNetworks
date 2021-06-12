#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: Softmaxp1
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi
from pprintpp import pprint as pp


def main() -> None:
    # we have to shape the samples
    # recall what this is [[0.1,0.5,0.4],[0.7,0.1,0.2] .... ] we are taking the
    # jth element from this list

    print(np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.008]]))

    softmax_output = np.array(([0.7, 0.1, 0.2])).reshape(-1, 1)

    pp(softmax_output)

    # we would have to use shape , because we are working with a side row

    print(np.diagflat(softmax_output))

    print("\n")
    print("S under i k dot s under i k ")
    print(np.dot(softmax_output, np.transpose(softmax_output)))

    print("\n \n:Partial dx of softmax outputis ")

    #  At this point we are doing jacobian matrix caculation
    print(x := (np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)))

    print("\n")
    print(
        "Applying np.maxiumum to see - what values are positive and what values are not "
    )
    print(np.maximum(0, x))

    print("This is a better understanding bellow, if you are still confused")

    print("\n\n\n\n")
    softmax_output = np.array(([0.7, 0.1, 0.2])).reshape(-1, 1)
    print(softmax_output.shape)
    print(softmax_output.T.shape)
    print(softmax_output, softmax_output.T)
    row_eyth = np.eye(softmax_output.shape[0])
    col_eyth = np.eye(np.transpose(softmax_output).shape[1])

    print(row_eyth == col_eyth)

    print(softmax_output * col_eyth)

    print(np.dot(softmax_output.T, col_eyth))

    print(2 * col_eyth)


if __name__ == "__main__":
    pi.install_traceback()

    main()


def code_examples() -> None:
    softmax_output = np.array([0.7, 0.1, 0.2])
    print("\n-----------")
    print("output of 0th index : ", softmax_output[0])
    print("Shape of the zeroth index :", softmax_output[0].shape)
    print("Np.eye of that shape ", np.eye(softmax_output.shape[0]))
    print("-----------\n")

    print(softmax_output * np.eye(softmax_output.shape[0]))

    print("this can be done faster if you would use the following commands \n")
