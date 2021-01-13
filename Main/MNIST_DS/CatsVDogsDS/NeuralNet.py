#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-31 Viv Sedov
#
# File Name: NeuralNet.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from frosch import hook
from tqdm import tqdm

import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        # this is some fake data
        x = torch.randn(50, 50).view(-1, 1, 50, 50)

        self._to_linear = None
        self.conv(x)

        # In this Scenario the order does very much matter

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # print(x[0].shape)
        if self._to_linear is None:

            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            # print(self._to_linear)
        return x

    def forward(self, x):
        x = self.conv(x)

        x = x.view(-1, self._to_linear)  # Flattern thoes weights
        # and other infomation . reshape it, to be flatterened .
        # We have to flat, but you have to know the number, when you start it, but you  would also
        # have to flat it  -1, to_linear to wht eever those values are
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Have an activation layer, but dont forget this .
        return F.softmax(x, dim=1)
        # See the difference between just returning x  , without softmax, and without an activation
        # Function , and we can just have fun , and mess around .


def main() -> None:
    ### This is the mainpart of the gpu - you have to define where that gpu does the procesisng

    ### --------------------------------------------------------
    device = torch.cuda.is_available()
    print(device)

    device = torch.device("cuda:0")
    print(device)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("On Gpu")
    else:

        device = torch.device("cpu")
        print("On Cpu ")
    ### --------------------------------------------------------
    # Encoder and decoder - and have that on two different gpus ?

    REBUILD_DATA = False
    net = Net().to(device)
    print(net)

    training_data = np.load("training_data.npy", allow_pickle=True)
    print(len(training_data))

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
    X = X / 255.0
    y = torch.Tensor([i[1] for i in training_data])

    VAL_PCT = 0.1  # lets reserve 10% of our data for validation
    val_size = int(len(X) * VAL_PCT)

    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]

    BATCH_SIZE = 512
    EPOCHS = 1000

    def train(net):
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                try:

                    # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
                    # print(f"{i}:{i+BATCH_SIZE}")
                    batch_X = train_X[i : i + BATCH_SIZE].view(-1, 1, 50, 50)
                    batch_y = train_y[i : i + BATCH_SIZE]
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    net.zero_grad()

                    outputs = net(batch_X)
                    loss = loss_function(outputs, batch_y)
                    loss.backward()
                    optimizer.step()  # Does the update
                except Exception as e:
                    pass
            print(f"Epoch: {epoch}. Loss: {loss *100}")
        print(loss)

    def test(net):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in tqdm(range(len(test_X))):
                real_class = torch.argmax(test_y[i]).to(device)
                net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]
                pc = torch.argmax(net_out)

                if pc == real_class:
                    correct += 1
                total += 1
                # This is one at a time .

        print(round(correct / total, 3))


    train(net)
    test(net)


if __name__ == "__main__":
    main()
