#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: ArgMaxDump
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi


def main() -> None:
    arg = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    print(np.argmax(arg, axis=1))


if __name__ == "__main__":
    pi.install_traceback()
    main()
