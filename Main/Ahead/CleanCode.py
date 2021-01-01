#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-29 Viv Sedov
#
# File Name: CleanCode.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import matplotlib.pyplot as plt
import nnfs
import numpy as np
from frosch import hook
from nnfs.datasets import spiral_data, vertical_data
from pprintpp import pprint as pp


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

    def accuracy(X, y) -> float:
        x = np.argmax(X, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        pp(x)
        pp(y)
        return np.mean(x == y)

    print(accuracy(X, y))


def random_optimization():
    X, y = vertical_data(samples=100, classes=3)
    # Using plt.scanenr instead of plt.imshow,

    # cmap - color map
    # C -> color sequence, this is not required - it allows us to seperate data that we have .

    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap="brg")
    # C = y ?
    # s  = size you moron ... s = 10, makes those dots small .
    # cmap = 'Im  guessing this is the type of color that we have '
    # x axis = X[:,0]
    # y axis = X[:,1]
    # plt.show()

    dense1 = DenseLayer(2, 3)
    activation1 = ActivationReLU()
    dense2 = DenseLayer(3, 3)
    activation2 = ActivationSoftMax()

    loss_function = Loss_CategoricalCrossentropy()

    # Create some variables to trqack the best loss and weights with teh biases

    lowest_loss = 99999999  # random var
    # Have coppy of weights and biases
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()

    best_dense2_weights = dense2.weights.copy()
    best_dense1_biases = dense2.biases.copy()

    """
    Loss to large value, and decrease it when a new lower loss is found, 
    and just coppying weights and bias, due to how python workds via oop  .
    copy() => is a function within our given parameters . 

    """

    for iteration in range(1000000):
        # for each weight and bias add on some very small value
        dense1.weights += 0.005 * np.random.randn(2, 3)
        dense1.biases += 0.005 * np.random.randn(1, 3)

        dense2.weights += 0.005 * np.random.randn(3, 3)
        dense2.biases += 0.005 * np.random.randn(1, 3)

        # push forwrd some extra value onwards
        dense1.forward(X)
        # But in this case we are manually changing those weights within our own range .

        activation1.forward(dense1.outputs)
        dense2.forward(activation1.outputs)

        # We have two layers here .
        activation2.forward(dense2.outputs)
        # now calculate the given loss that we have .

        # Calculate the orignal loss from softmax output .
        loss = loss_function.calculate(activation2.outputs, y)
        arger = np.argmax(activation2.outputs, axis=1)
        accuracy = np.mean(arger == y)

        if loss < lowest_loss:
            print(
                "New set of weights found within the given iteration {} loss {} acc {} ".format(
                    iteration, loss, accuracy
                )
            )
            # that given copy, bcomes what ever the dense value oof the weights are now

            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()

            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_lost = loss

            # rever to the previous weight and changes that it had before .
            # if thta is not the case, revert back to the previous values it had before hand .

        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()

            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()

    print(lowest_lost)

    # In this scenario, when we are optimising values within the given bias

    # It just seems, stating those values, to what ever our original copy was
    """


    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()

    best_dense2_weights = dense2.weights.copy()
    best_dense1_biases = dense2.biases.copy()


    pretty much what we are doing is stating that, we have this best dense which is the orignal, 
    we stat eback to what ever we have at first 
    or we continue onewards and prpaire for loss 
    """


if __name__ == "__main__":
    hook()

    # This does a defualt data type, for us, which is rather nice .
    nnfs.init()
    # main()
    random_optimization()
