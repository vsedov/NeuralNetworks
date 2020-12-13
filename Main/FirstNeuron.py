#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-12-13 Viv Sedov
#
# File Name: FirstNeuron.py
# Distributed under terms of the MIT license.
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import better_exceptions
import numpy as np
from frosch import hook
from pprintpp import pprint as pp

better_exceptions.hook()
hook()


def firstNeuron() -> np:
    x = np.array([1, 2])
    pp(x)


def main() -> None:
    firstNeuron()


if __name__ == "__main__":
    main()