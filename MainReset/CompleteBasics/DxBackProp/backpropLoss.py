#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: backpropLoss
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi
from pprintpp import pprint as pp


def main() -> None:

    dvalues = np.array([[1, 2, 3, 4, 5], [4, 3, 1, 4, 19]])

    samples = np.array([1])
    print(samples.shape)

    # if this is sparse other wise we do not have to do this step
    print(np.eye(len(dvalues))[samples])

    samples = np.array([[1, 0, 1, 0, 1], [0, 0, 0, 0, 1]])

    dinputs = -samples / dvalues

    dinputs = dinputs / len(dvalues)

    print("Overall dinputs after partial dx is done  - y  / yhat / len(yhat)")
    pp(dinputs)


if __name__ == "__main__":
    pi.install_traceback()
    main()
