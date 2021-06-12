#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: Softmax_2
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi
from pprintpp import pprint as pp


def method_1(softmax_output: np.ndarray, s_jk: np.ndarray) -> np.ndarray:
    return s_jk + np.dot(-softmax_output, softmax_output.T)


def method_2(softmax_output: np.ndarray, s_jk: np.ndarray) -> np.ndarray:
    return s_jk - np.dot(softmax_output, softmax_output.T)


def main() -> None:

    softmax_output = np.array(([0.7, 0.1, 0.2])).reshape(-1, 1)
    s_jk = np.diagflat(softmax_output)

    pp(method_1(softmax_output, s_jk))
    print("\n")
    pp(method_2(softmax_output, s_jk))


if __name__ == "__main__":
    pi.install_traceback()
    main()
