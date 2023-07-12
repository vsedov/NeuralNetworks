#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-29 Viv Sedov
#
# File Name: CleanCode.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import nnfs
import numpy as np
from frosch import hook
from nnfs.datasets import spiral_data


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # This is just random generated results
        self.biases = np.zeros((1, n_neurons))
        # If it is dying, but you can change that to a non zero

    def forward(self, inputs: list) -> np:
        self.outputs = np.dot(inputs, self.weights) + self.biases

    # Very simple, but very nice to do as well .


class ActivationReLU:
    def forward(self, inputs: list) -> np:
        self.outputs = np.maximum(0, inputs)


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
        return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred: list, y_true: list) -> np:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-07, 1 - 1e-07)

        # Looking at the target values

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidence)


def main() -> None:

    X, y = spiral_data(samples=100, classes=3)  # This is your data

    dense1 = DenseLayer(2, 3)  # So inputs are 2
    dense2 = DenseLayer(3, 3)
    # ^ Because classes is three
    activation1 = ActivationReLU()
    activation2 = ActivationReLU()
    loss_function = Loss_CategoricalCrossentropy()

    dense1.forward(X)
    activation1.forward(dense1.outputs)

    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)
    loss = loss_function.calculate(activation2.outputs, y)
    print("Loss : ", loss)

    pred = np.argmax(activation2.outputs, axis=1)

    print(pred)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(pred == y)
    print(accuracy)


if __name__ == "__main__":
    hook()

    # This does a defualt data type, for us, which is rather nice .
    nnfs.init()
    main()
