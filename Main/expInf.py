#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-25 Viv Sedov
#
# File Name: expInf.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook


def main() -> None:
    print(np.exp(-np.inf), np.exp(0))


if __name__ == "__main__":
    hook()
    main()
