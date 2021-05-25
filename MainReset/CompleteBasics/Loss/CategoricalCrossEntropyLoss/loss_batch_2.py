#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: loss_batch_2
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi
from pprintpp import pprint as pp


def log_loss_batch(softmax_output: list, class_target: list) -> None:
    print(
        "Without using the loss formula:",
        softmax_output[range(len(softmax_output)), class_target],
        sep="\n",
    )

    print("\n")
    pp(["Class Target:", class_target])
    print("\n")
    pp(softmax_output)

    print("\n")
    # Without using sumunation here , though you can if you want to
    neg_log = (-np.log(softmax_output[range(len(softmax_output)), class_target]),)
    average_loss = np.mean(neg_log)
    print(average_loss)


def main() -> None:
    class_targets = [0, 1, 1]
    softmax_output = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.008]])
    log_loss_batch(softmax_output, class_targets)


if __name__ == "__main__":
    pi.install_traceback()
    main()
