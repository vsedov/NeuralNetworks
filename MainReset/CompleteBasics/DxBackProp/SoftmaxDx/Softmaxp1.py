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
    print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))


    

if __name__ == "__main__":
    pi.install_traceback()

    main()


def code_examples()->None:
    print("\n-----------")
    print("output of 0th index : ", softmax_output[0])
    print("Shape of the zeroth index :", softmax_output[0].shape)
    print("Np.eye of that shape ", np.eye(softmax_output.shape[0]))
    print("-----------\n")

    print(softmax_output * np.eye(softmax_output.shape[0]))

    print("this can be done faster if you would use the following commands \n")
