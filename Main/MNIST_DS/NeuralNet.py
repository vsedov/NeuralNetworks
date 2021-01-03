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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from frosch import hook
from pprintpp import pprint as pp
from torchvision import datasets, transforms

logging.basicConfig(filename="LogFile.log", level=logging.INFO)
# Inheriting all values within the nnmodules -
# Init running from super, and we do super init .

torch.cuda.set_device(0)


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
        inputs = F.relu(self.fc2(inputs))
        inputs = F.relu(self.fc3(inputs))
        inputs = self.fc4(inputs)
        return F.softmax(inputs, dim=1)

    # Pretty certain dim is axis - like numpy and axis = 1

    # Single Input chage
def main() -> None:

    train = datasets.MNIST(
        "",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test = datasets.MNIST(
        "",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    # This was annoying as this took me more time to figure out what really is going on
    trainset = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

    net = Net()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    EPOCHS = 30  # this is for full data parsing

    for EPOCHS in range(EPOCHS):
        for data in trainset:
            # data is a batch of featuresets and labels
            X, y = data
            # Container that contains 10 feature sets

            net.zero_grad()  # Is this stating that this is the train set ?

            output = net(
                X.view(-1, 28 * 28)
            )  # this is the image compression that we have to do ,.

            loss = F.nll_loss(output, y)

            loss.backward()
            # Backprop MAGIC
            # Iterate net.parameters and distribuate it
            optimizer.step()  # This changes the weights for us .
        print(loss)

    correct = 0
    total = 0

    # no_grad => within that training mode .

    with torch.no_grad():
        for data in trainset:
            X, y = data
            output = net(X.view(-1, 28 * 28))  # Shapping that data
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy :", correct / total)
    print(X)


if __name__ == "__main__":
    hook()
    main()
