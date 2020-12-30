#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-29 Viv Sedov
#
# File Name: tests.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook
from pprintpp import pprint as pp


def main() -> None:
    x = np.array([[1.00, 0, 0], [1, 0, 0], [1, 0, 0]])
    target = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    x = np.clip(x, 1e-7, 1 - 1e-7)

    sumer = np.sum(x * target, axis=1)

    neg_log = -np.log(sumer)
    mean = np.mean(neg_log)


if __name__ == "__main__":
    hook()
    main()
