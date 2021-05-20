#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: log
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi


def main() -> None:
    print("b=ln(x) -> e**b=x ")
    b = 5.2
    print("Bth value: ", b)
    print(exp_val := (np.log(b)))
    print(np.exp(exp_val))


if __name__ == "__main__":
    pi.install_traceback()
    main()
