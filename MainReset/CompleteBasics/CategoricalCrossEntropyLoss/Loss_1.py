#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: Loss_1
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import math

import nnfs
import pyinspect as pi


def main() -> None:
    softmax_output = [0.7, 0.1, 0.2]
    target_output = [1, 0, 0]

    # -----------------------------
    # Doing this with list comp and using the formulation for everything
    # $\sum(y_i_k \cdot \log(Y_i_k))$ log = ln()
    # -∑ (yₖ∙log(Yₖ))
    cross_entropy = [
        math.log(softmax_output[i]) * target_output[i]
        for i in range(len(softmax_output))
    ]
    print(-sum(cross_entropy))

    # ----------------
    # Notice how this is the same as the following formula
    # $-log(Y_k_i) $

    loss = -(math.log(softmax_output[0]) * target_output[0])
    print(loss)


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main()
