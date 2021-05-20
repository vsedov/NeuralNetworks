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


def main() -> None:
    softmax_output = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.008]])
    class_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main()
