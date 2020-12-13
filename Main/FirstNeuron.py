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

better_exceptions.hook()
hook()


def firstNeuron() -> np:
    # MultiLyaer perceptron model :  For the first level
    # Words can be long .
    # Every neuron has a unique connection to a previous connection .
    # The output
    inputs = [1.2, 2.5, 2.1]  # Unique inputs and outputs from previous layer
    weights = [3.1, 2.1, 8.9]  # Each input, and a weight for each input
    biases = 3  # is the ammount that we have -

    output = (
        inputs[0] * weights[0]
        + inputs[1] * weights[1]
        + inputs[2] * weights[2]
        + biases
    )
    print(output)  # In this case

    # this would be the first output for first neuron that we made .

    # For the most part


def main() -> None:

    firstNeuron()


if __name__ == "__main__":
    main()

    """
    we break down every step :
    input = Inputs for the user .
    weights, in this case we have some sort of line so this output will be the
    input for the next layer
    bias is the value that we have atm .
    """
