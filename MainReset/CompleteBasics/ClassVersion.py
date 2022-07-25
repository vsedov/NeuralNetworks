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
from icecream import ic
from nnfs.datasets import spiral_data  # import nnfs


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
        """Used for BackProp"""
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)


class ActivationRelu:

    def forward(self, inputs: np.ndarray) -> None:
        self.output = np.maximum(0, inputs)
        """For BackProp"""
        self.inputs = inputs

    def backward(self, dvalues: np.ndarray) -> None:
        """Used for BackProp"""
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftMax:

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # This would be when you are parsing data down - without using the chain
    # rule
    def backward(self, dvalues: np.ndarray) -> None:
        # An Empty array
        self.dinputs = np.empty_like(dvalues)
        # print("\n all dvalues\n", dvalues, "\n")
        self.jacobian_matrix = []
        print("----- Starting loop for Jacobian Matrix ----\n")
        for index, (single_output,
                    single_dvalues) in enumerate(zip(self.output, dvalues)):
            # print("Single dvalues ", single_dvalues, "\n")

            self.dinputs[index] = np.dot(
                self.jacobian(single_output.reshape(-1, 1)), single_dvalues)

            self.jacobian_matrix.append(
                self.jacobian(single_output.reshape(-1, 1)))
        ic(self.jacobian_matrix)

    @classmethod
    def jacobian(cls, single_output: np.ndarray) -> np.ndarray:
        return np.diagflat(single_output) - np.dot(single_output,
                                                   single_output.T)


class Loss:
    # A accuracywraper could be used here
    def accuracy(self, output: np.ndarray, y: np.ndarray) -> str:
        # Index of all the highest values in axis=1 row form
        """Convert infomation to sparse infomation and create a loss out of argmax"""
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        #                   Predictions :
        sample = np.argmax(output, axis=1)
        # Accuracy
        # return f"Acc: {round(float(np.mean(sample == y, keepdims=True)), 5)} %"
        return np.mean(sample == y)

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


class ActivationSoftMaxCCELoss(Loss):

    def __init__(self):
        self.activation = ActivationSoftMax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.caculate(self.output, y_true)

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        samples = len(dvalues)
        # ic("Given values ", dvalues)

        # Recall jacobian matrix can be reshaped into 1 dimentional
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples


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

        # shape -> 2 => [1,0]
        # shape -> 1 => [[1,1],[0.1]]
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidence)

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """BackProp"""
        # Number of samples

        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector - Len in this case
        # very imporant - Rather jaring that i fucked this part up before
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class OptimizerSGD:

    def __init__(self,
                 learning_rate: float = 1.0,
                 decay: float = 0.0,
                 momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer: LayerDense) -> None:
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.bias)

            weight_updates = (self.momentum * layer.weight_momentums -
                              self.current_learning_rate * layer.dweights)
            layer.weight_momentums = weight_updates

            bias_updates = (self.momentum * layer.bias_momentums -
                            self.current_learning_rate * layer.dbias)
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbias

        layer.weights += weight_updates
        layer.bias += bias_updates

    def post_update_params(self) -> None:
        self.iterations += 1


def main() -> None:

    # categorical_data woudl be the classification that you are trying to get
    X, categorical_data = spiral_data(samples=100, classes=3)  # noqa:ignore

    dense_1 = LayerDense(2, 3)

    activation_1_relu = ActivationRelu()

    dense_2 = LayerDense(3, 3)

    loss_activation_ssl = ActivationSoftMaxCCELoss()

    dense_1.forward(X)

    activation_1_relu.forward(dense_1.output)

    dense_2.forward(activation_1_relu.output)

    loss = loss_activation_ssl.forward(dense_2.output, categorical_data)
    soft_max_output = loss_activation_ssl.output
    accuracy = loss_activation_ssl.accuracy(dense_2.output, categorical_data)

    print("Loss: ", loss)
    print(accuracy)
    print("Softmax output ", np.sum(soft_max_output[:5], axis=1,
                                    keepdims=True))

    # Doing the backward pass

    # Loss and softmax with or together
    loss_activation_ssl.backward(loss_activation_ssl.output, categorical_data)
    dense_2.backward(loss_activation_ssl.dinputs)
    activation_1_relu.backward(dense_2.dinputs)
    dense_1.backward(activation_1_relu.dinputs)

    print("\n-------------\nDense_1 Weights and Bias ---------\n")
    print("Weights \n", dense_1.dweights, "\n")
    print("Bias \n", dense_1.dbias, "\n")

    print("\n-------------\nDense_2 Weights and Bias ---------\n")
    print("Weights \n", dense_2.dweights, "\n")
    print("Bias \n", dense_2.dbias, "\n")


def main_version_2() -> None:
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)  # noqa :ignore

    dense1 = LayerDense(2, 64)

    activation1 = ActivationRelu()

    dense2 = LayerDense(64, 3)
    loss_activation = ActivationSoftMaxCCELoss()

    # Create optimizer

    # In this case 0.9 performs better for some reason
    # optimizer = OptimizerSGD(learning_rate=0.86000)
    optimizer = OptimizerSGD(decay=1e-3, momentum=0.9)
    # Train in loop
    for epoch in range(10001):

        dense1.forward(X)

        activation1.forward(dense1.output)

        dense2.forward(activation1.output)

        loss = loss_activation.forward(dense2.output, y)

        accuracy = loss_activation.accuracy(loss_activation.output, y)

        # predictions = np.argmax(loss_activation.output, axis=1)
        # if len(y.shape) == 2:
        #     y = np.argmax(y, axis=1)

        # accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f"epoch: {epoch}, " + f"acc: {accuracy:.3f}, " +
                  f"loss: {loss:.3f}, " +
                  f"lr: {optimizer.current_learning_rate}")

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()


def softmaxloss() -> None:
    softmax_output = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4],
                               [0.02, 0.9, 0.008]])
    class_targets = np.array([0, 0, 1])
    softmax_loss = ActivationSoftMaxCCELoss()
    softmax_loss.backward(softmax_output, class_targets)
    dvalues1 = softmax_loss.dinputs

    print("\n-------------------\n")

    activation = ActivationSoftMax()
    activation.output = softmax_output
    loss = LossCategoricalCrossEntropy()

    loss.backward(softmax_output, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs

    print("From 1- Combined values \n", dvalues1)
    print("\n-------------------\n")
    print("From 2 - Sep loss with softmax \n", dvalues2)

    # For the basis of infomation , i made this output its infomation into
    # a string , as i thought it would be nicer with respects to the given
    # output
    print(softmax_loss.accuracy(softmax_output, class_targets))


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main_version_2()
    # softmaxloss()
    # softmaxloss()
