#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-28 Viv Sedov
#
# File Name: NeuralNet.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import logging

import numpy as np
# nn Is for oop -> nn.functional and nn from function - both are a bit weird
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from frosch import hook
from pprintpp import pprint as pp

logging.basicConfig(filename="LogFileForTorchnet.log", level=logging.INFO)

# Inheriting all values within the nnmodules -
# Init running from super, and we do super init .


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)  # 28 x 28 - we parse that as the input

        self.fc2 = nn.Linear(64, 64)  #  what ever output = input for next layer,

        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 10)
        # Output is how ever many classifiers that you have .

    def forward(self, inputs: torch.Tensor) -> nn:
        # You can put logic in here  - like
        # if weather == sunny : do some other layers .
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = F.relu(self.fc3(inputs))
        inputs = self.fc4(inputs)
        return F.log_softmax(inputs, dim=1)

    # Pretty certain dim is axis - like numpy and axis = 1


def main() -> None:
    net = Net()
    X = torch.rand((28, 28))
    X = X.view(-1, 28 * 28)
    print(type(X))
    output = net.forward(X)
    print(output)


if __name__ == "__main__":
    hook()
    main()
