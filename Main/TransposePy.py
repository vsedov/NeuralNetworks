#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-23 Viv Sedov
#
# File Name: TransposePy.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook

np.random.seed(0)


class DensLayer:
    def __init__(self, nInputs, nNeurons):
        # We dont have to transpose everytime, when you ahve control ove rthe first section self.weights = 0.01 * np.random.randn( nInputs, nNeurons)  # This creates the shape that we want to pass , Size of input : Size of Neurons
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        self.bias = np.zeros(
            (1, nNeurons)
        )  # Shape is 1 x By how ever many Neruons that you have .
        # This would have to be Shape as a parameter ie, this would have to be a tuple . Parem are the shape it self ....

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


def tester() -> np:
    X = [
        [1, 2, 3, 2.5],
        [2, 5, -1, 2],
        [-1.5, 2.7, 3.3, -0.8],
    ]  # Assuming this is the input data
    # Nneurons is what ever you want

    layer1 = DensLayer(4, 5)
    layer2 = DensLayer(5, 2)
    layer3 = DensLayer(2, 10)

    layer1.forward(X)

    layer2.forward(layer1.output)

    layer3.forward(layer2.output)

    print(layer3.output)


def main() -> None:

    tester()


if __name__ == "__main__":
    hook()
    main()

    """
    Smaller values are better, and teh infomation, have a rand between neg 1 and pos 1
    the weights are, amount of inputs, so like say you have some input,  and you have nNeurons for that input . 
   """
