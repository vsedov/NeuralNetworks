#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-20 Viv Sedov
#
# File Name: batchAndLayers.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from pprintpp import pprint as pp
from icecream import ic


def calCu() -> np:  # Shape Error

    inputs = [1, 2, 3, 2.5]

    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, 0.8, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    bias = [2, 3, 0.5]  # We are not changing anything else here

    # Index of one - dot needs to match Index Zero
    pp(np.dot(weights, inputs) + bias)


def calCuFixShape() -> np:  # Shape Error

    inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    bias = [2, 3, 0.5]  # We are not changing anything else here

    # Index of one - dot needs to match Index Zero

    # Convert weights to numpy array

    # So this is what you would do

    output = np.dot(inputs, np.array(weights).T) + bias

    pp(output)
    ic(len(inputs), len(weights))

    print("Going to the second layer \n")

    # In this case, the given layer, : has to be the same as the amount of bias and weights
    # we had before

    weights2 = [
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13],
    ]

    bias2 = [-1, 2, -0.5]  # We are not changing anything else here

    layer1_Output = np.dot(inputs, np.array(weights).T) + bias

    # Layer1_Ouput = becomes into layer1 inputs basiclly if that makes sense

    layer2_Output = np.dot(layer1_Output, np.array(weights2).T) + bias2

    pp(layer2_Output)

    # This can become rather long


def classer():
    X = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]  # Input data is always an X

    # Defining the two latent space

    # You can normalise this data as well and you can scale it -1 and +1
    ic("In CLASS ")

    np.random.seed(0)

    layer1 = Layer_Dense(4, 5)  # This is number of neurons .
    # the input has to be the same as the layer value
    layer2 = Layer_Dense(5, 6)

    layer1.forward(X)
    print(layer1.output)

    layer2.forward(layer1.output)


class Layer_Dense:  # making a new neural networks := Weights First -1<= x <=1
    def __init__(self, n_inputs, n_neurons):
        #
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons
        )  # we need to know the shape Size fo input and How many Neurons
        # Size of a single sample so 4 ^
        self.biases = np.zeros(
            (1, n_neurons)
        )  # will be 1 by how many neurons that you have - Needs to be a tuple of those two
        # 1 * how many neurons do you have ?, you do not have to do it for the ammount of inputs.
        # this allows us to avoid doing a transpose every time .
        # ic(self.biases, self.biases.shape)

    def forward(self, inputs):  # This could be the training data

        self.output = np.dot(inputs, self.weights) + self.biases


def main() -> None:
    # calCu()
    # print("\n")

    calCuFixShape()
    print("Changing this into a class \n")
    classer()


if __name__ == "__main__":
    main()
