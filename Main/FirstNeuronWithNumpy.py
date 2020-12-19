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
    inputs = [1, 2, 3, 2.5]
    weights = [0.2, 0.8, -0.5, 1]
    biases = 2
    pp(np.dot(inputs, weights) + biases)

    print("Those values that we have here are defined by ")
    print("{}".format(inputs))

def secondNeuron() -> np:
    pass 

def main() -> None:

    firstNeuron()


if __name__ == "__main__":
    main()
