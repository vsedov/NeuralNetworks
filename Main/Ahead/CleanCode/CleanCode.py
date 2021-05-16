#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-29 Viv Sedov
#
# File Name: CleanCode.py
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import nnfs
import numpy as np
from frosch import hook
from nnfs.datasets import spiral_data
from pprintpp import pprint as pp


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # This is just random generated results
        self.biases = np.zeros((1, n_neurons))
        # If it is dying, but you can change that to a non zero

    def forward(self, inputs: list) -> np:
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.dweights.T)


class ActivationReLU:
    def forward(self, inputs: list) -> np:
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftMax:
    def forward(self, inputs: list) -> np:
        expVal = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Cool this works - Take Value away and keeep that dimension .
        # This prevents overflow error

        probability = expVal / np.sum(expVal, axis=1, keepdims=True)

        self.outputs = probability


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        # This function above would calculate the loss

        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred: list, y_true: list) -> np:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-07, 1 - 1e-07)

        # Looking at the target values

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        cross_entropy = -np.log(correct_confidence)
        return cross_entropy


def main() -> None:

    X, y = spiral_data(samples=100, classes=3)  # This is your data

    dense1 = DenseLayer(2, 10)  # So inputs are 2
    dense2 = DenseLayer(10, 5)
    dense3 = DenseLayer(5, 10)
    dense4 = DenseLayer(10, 3)
    # ^ Because classes is three
    activation1 = ActivationReLU()
    activation2 = ActivationReLU()
    activation3 = ActivationReLU()
    activation4 = ActivationSoftMax()

    loss_function = Loss_CategoricalCrossentropy()

    dense1.forward(X)
    activation1.forward(dense1.outputs)

    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)

    dense3.forward(activation2.outputs)
    activation3.forward(dense3.outputs)

    dense4.forward(activation3.outputs)
    activation4.forward(dense4.outputs)

    loss = loss_function.calculate(activation4.outputs, y)
    print("Loss : ", loss)

    # This just compares those values with argmax
    def accuracy(X, y) -> float:
        x = np.argmax(X, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        pp(x)
        pp(y)
        return np.mean(x == y)

    print(accuracy(X, y))


if __name__ == "__main__":
    hook()

    # This does a defualt data type, for us, which is rather nice .
    nnfs.init()
    main()
