#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: Mom
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import pyinspect as pi


def momentum_test(
    start_rate: int = 1.0, decay_rate: int = 0.01, step: int = 0.5
) -> None:
    learning_rate = start_rate * (1.0 / (1 + decay_rate * step))

    print(learning_rate)


def main() -> None:

    pass


if __name__ == "__main__":
    pi.install_traceback()
    main()
