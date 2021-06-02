#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: main_network
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import nnfs
import numpy as np
import pyinspect as pi
from nnfs.datasets import spiral_data

nnfs.init()


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        # weights are made for you - transpose not needed
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> None:
        self.output = np.dot(inputs, self.weights) + self.bias
        """Used For backprop"""
        self.inputs = inputs

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)


class ActivationRelu:
    def forward(self, inputs: np.ndarray) -> None:
        self.output = np.maximum(0, inputs)
        """For BackProp"""
        self.inputs = inputs

    def backward(self, dvalues: np.ndarray) -> None:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftMax:
    def forward(self, inputs: np.ndarray) -> None:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Loss:
    def accuracy(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Index of all the highest values in axis=1 row form

        """Convert this infomation to sparse infomation"""
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        #                   Predictions :
        sample = np.argmax(output, axis=1)
        # Accuracy
        return np.mean(sample == y, keepdims=True)

    def caculate(self, outputs: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        # DataLoss
        return np.mean(sample_losses)


# Each main loss function will be using the loss caculation
# Caculate will be done after teh loss it self was found
class LossCategoricalCrossEntropy(Loss):
    """
    CrossEntropyLoss formual
        x = one hot
        Y = Predicted Values
        L(x,Y) => -∑ (x∙ln(Y))
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
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

        return -np.log(correct_confidence)


def main() -> None:

    # y woudl be the classification that you are trying to get
    input_data, categorical_data = spiral_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationRelu()

    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftMax()

    dense1.forward(input_data)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss_function = LossCategoricalCrossEntropy()
    loss = loss_function.caculate(activation2.output, categorical_data)
    print("Loss : ", loss)

    accuracy = loss_function.accuracy(activation2.output, categorical_data)
    print("Accuracy : ", accuracy)


if __name__ == "__main__":
    pi.install_traceback()
    main()
