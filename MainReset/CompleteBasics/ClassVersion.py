#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: ClassVersion
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import nnfs
import numpy as np
import pyinspect as pi
from nnfs.datasets import spiral_data
from pprintpp import pprint as pp

nnfs.init()


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        # weights are made for you - transpose not needed
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.output = np.dot(inputs, self.weights) + self.bias


class ActivationRelu:
    def forward(self, inputs: np.ndarray):
        self.output = np.maximum(0, inputs)


class ActivationSoftMax:
    def forward(self, inputs: np.ndarray):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Loss:
    def accuracy(self, output: np.ndarray, y: np.ndarray):
        # Index of all the highest values in axis=1 row form

        """Convert this infomation to sparse infomation"""
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        #                   Predictions :
        sample = np.argmax(output, axis=1)
        accuracy = np.mean(sample == y, keepdims=True)
        return accuracy

    def caculate(self, outputs: np.ndarray, y: np.ndarray):
        """
        Caculate Loss mean

        Takes loss method and does forward prse
        np.mean(self.forward(x))

        Parameters
        ----------
        outputs : np.ndarray
            Softmax Data
        y : np.ndarray
            One hot encoded value

        Returns : mean loss
        """
        sample_losses = self.forward(outputs, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# Each main loss function will be using the loss caculation
# Caculate will be done after teh loss it self was found
class LossCategoricalCrossEntropy(Loss):
    """
    CrossEntropyLoss formual
        x = one hot
        Y = Predicted Values
        L(x,Y) => -∑ (x∙ln(Y))
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        y_pred = Y :: y_true = x
        """
        # Number of samples within the batch
        samples = len(y_pred)

        # Clip both sides to note drag mean value, only applied for
        # 1 and 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        pp(correct_confidence[:10])
        negative_likehood = -np.log(correct_confidence)
        return negative_likehood


def main() -> None:

    # y woudl be the classification that you are trying to get
    X, y = spiral_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationRelu()

    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftMax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(
        "Without Checking if they added up to one softmax: \n",
        limited := (activation2.output[:5]),
    )
    print(
        "\n, Checking that they would all added up to one: \n",
        np.sum(limited, axis=1, keepdims=True),
    )
    print("\n")

    loss_function = LossCategoricalCrossEntropy()
    loss = loss_function.caculate(activation2.output, y)
    print("Loss : ", loss)

    accuracy = loss_function.accuracy(activation2.output, y)
    print("Accuracy : ", accuracy)


if __name__ == "__main__":
    pi.install_traceback()
    main()
