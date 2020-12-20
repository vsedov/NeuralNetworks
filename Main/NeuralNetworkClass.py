#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-20 Viv Sedov
#
# File Name: NeuralNetworkClass.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook

"""
For this case we have, 
n_neurons is the next output of neurons that we have
n_inputs is the length of how ever many inputs we get 


so for the next given value the inputs would allways be the n_neurons 

"""


class Dense_Layer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons
        )  # This is just random generated results
        self.biases = np.zeros((1, n_neurons))

    def forall(self, inputs: list) -> np:
        self.outputs = np.dot(inputs, self.weights) + self.biases


def main() -> np:

    X = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]  # Input data is always an X

    W_1 = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    x = 10  # This can be any other value or number that you want
    layer_1 = Dense_Layer(4, x)

    layer_1.forall(X)
    # print(layer_1.outputs)

    layer_2 = Dense_Layer(x, 2)  # This has to be what ever it had before

    layer_2.forall(layer_1.outputs)
    print(layer_2.outputs)


if __name__ == "__main__":
    hook(theme="fruity")
    main()
