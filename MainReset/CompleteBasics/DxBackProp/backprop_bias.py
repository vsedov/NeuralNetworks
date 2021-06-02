#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: backprop_bias
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi


def main() -> None:

    # this is the passed in gradient from the next layer, this is just som
    # ebasic values and infomation that we will be using for now , though this does
    # get quite complex later on

    # though these are the biases andf infomation , because of how back prop
    # works , the partial dx in relation to its self, will always be one , so
    # when you do the chain rule , you will just be multiplying the collumns in
    # this case
    dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    # this would become 1 1 1
    bias = np.array([[2, 3, 0.5]])

    dbias = np.sum(dvalues, axis=0, keepdims=True)

    print(dbias)


if __name__ == "__main__":
    pi.install_traceback()
    main()
