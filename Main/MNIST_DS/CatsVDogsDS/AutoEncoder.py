#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2021-01-07 Viv Sedov
#
# File Name: AutoEncoder.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import logging
from dataclasses import dataclass

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
from tqdm import tqdm

torch.cuda.set_device(0)
logging.basicConfig(filename="AutoEncoder.log", level=logging.DEBUG)


class AutoEncoder(nn.Module):
    def __init__(self, inputs):
        super().__init__()

        # This is quite basic but why not, just go with teh flow
        self.encoder_hidden = nn.Linear(inputs, 128)
        self.encoder_output = nn.Linear(128, 64)
        self.encoder_output1 = nn.Linear(64, 32)

        self.decoder_output1 = nn.Linear(32, 64)
        self.decoder_hidden = nn.Linear(64, 128)
        self.decoder_output = nn.Linear(128, inputs)

    # Another qite basic forward pass
    def forward(self, inputs: torch.Tensor) -> nn:
        inputs = F.leaky_relu(self.encoder_hidden(inputs))
        inputs = F.leaky_relu(self.encoder_output(inputs))
        inputs = torch.sigmoid(self.encoder_output1(inputs))

        inputs = F.leaky_relu(self.decoder_output1(inputs))
        inputs = F.leaky_relu(self.decoder_hidden(inputs))
        inputs = F.leaky_relu(self.decoder_output(inputs))
        return inputs


def main() -> None:
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

    REBUILD_DATA = False
    net = AutoEncoder().to(device)
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

    BATCH_SIZE = 256
    EPOCHS = 10

    net = AutoEncoder(inputs=784).to(device)

    # Optimiser, which im currently coding a manual version of it .

    optimiser = optim.Adam(net.parameters(), lr=1e-3)

    loss_function = nn.MSELoss()

    trainset = torch.utils.data.DataLoader(
        train, batch_size=512, shuffle=True, num_workers=4, pin_memory=True
    )
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

    test_examples = None

    epoch = 20

    for epoch in range(epoch):
        loss_count = 0
        for data, _ in trainset:
            data = data.view(-1, 784)
            data = data.to(device)

            optimiser.zero_grad()  # What this should be linked to ?
            output = net(data).to(device)

            train_loss = loss_function(output, data)
            train_loss.backward()
            optimiser.step()
            loss_count += train_loss.item()

        print(
            f"epoch {epoch +1 } / {epoch} loss = {loss_count} and given loss is {train_loss}"
        )

    test_examples = None

    with torch.no_grad():
        for data in testset:
            data = data[0]
            test_examples = data.view(-1, 784).to(device)
            recon = net(test_examples)
            break

    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for i in range(number):
            # display original
            plotter = plt.subplot(2, number, i + 1)
            plt.imshow(test_examples[i].cpu().numpy().reshape(28, 28))
            plt.gray()
            plotter.get_xaxis().set_visible(False)
            plotter.get_yaxis().set_visible(False)

            # display reconstruction
            plotter = plt.subplot(2, number, i + 1 + number)
            plt.imshow(recon[i].cpu().numpy().reshape(28, 28))
            plt.gray()
            plotter.get_xaxis().set_visible(False)
            plotter.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    hook()
    main()
