#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: learning_rate_decay
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import pyinspect as pi


def decay(step: int) -> None:
    start_learning_rate = 1.0
    decay_rate = 0.1

    learning_rate = start_learning_rate * (1.0 / (1 + decay_rate * step))

    print(learning_rate)


def main() -> None:

    for step in range(20):
        decay(step)


if __name__ == "__main__":
    pi.install_traceback()
    main()
