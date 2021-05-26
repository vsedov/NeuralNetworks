#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: Bad_opt_example
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import matplotlib.pyplot as plt
import nnfs
import numpy as np
import pyinspect as pi
from nnfs.datasets import vertical_data

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

        negative_likehood = -np.log(correct_confidence)
        return negative_likehood


def graph_view(X: np.ndarray, y: np.ndarray):

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="brg")
    plt.show()


def main() -> None:
    X, y = vertical_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationRelu()

    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftMax()

    loss_function = LossCategoricalCrossEntropy()

    """
    Create some variables to track the best loss and the associated weights and biases
    """

    lowest_loss = 999999

    # Copy the dense layers
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.bias.copy()

    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.bias.copy()

    """Coppy: full copy and reference of the object"""

    for iteration in range(10000):

        # Adding Derivatives, and pointers to push forward
        # This allows the accuracy to be more / loss to be lower
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.bias += 0.05 * np.random.randn(1, 3)

        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.bias += 0.05 * np.random.randn(1, 3)

        # This is the forward parse  for training the infomation through the
        # data
        dense1.forward(X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Perfrom a forward pass through the activation function
        # such that you would then caculate the given loss

        # Loss

        loss = loss_function.caculate(activation2.output, y)

        accuracy = loss_function.accuracy(activation2.output, y)

        # If the loss is smaller than the lowest_loss, than we have to print and
        # save the biases on the side

        if loss < lowest_loss:
            print(
                "New set of weights found,  iteration :",
                iteration,
                "Loss : ",
                loss,
                "Accuracy : ",
                accuracy,
            )

            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.bias.copy()

            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.bias.copy()

            lowest_loss = loss

    # print("Given dataset is :\n")
    # graph_view(X, y)


if __name__ == "__main__":
    pi.install_traceback()
    main()
