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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from frosch import hook
from torchvision import datasets, transforms

torch.cuda.set_device(0)
logging.basicConfig(filename="AutoEncoder.log", level=logging.DEBUG)


class AutoEncoder(nn.Module):
    def __init__(self, inputs: np.ndarray):
        super().__init__()

        # This is quite basic but why not, just go with teh flow
        self.encoder_hidden = nn.Linear(inputs, 128)
        self.encoder_output = nn.Linear(128, 64)
        self.encoder = nn.Linear(64, 32)

        self.encoder_output1 = nn.Linear(64, 32)
        self.decoder_output1 = nn.Linear(32, 64)

        self.decoder_hidden = nn.Linear(64, 128)
        self.decoder_output = nn.Linear(128, inputs)

    # Another qite basic forward pass
    def forward(self, inputs: torch.Tensor) -> nn:
        inputs = f.leaky_relu(self.encoder_hidden(inputs))
        inputs = f.leaky_relu(self.encoder_output(inputs))

        inputs = torch.sigmoid(self.encoder_output1(inputs))
        inputs = torch.sigmoid(self.decoder_output1(inputs))

        inputs = f.leaky_relu(self.decoder_hidden(inputs))
        return f.leaky_relu(self.decoder_output(inputs))


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

    train = datasets.MNIST(
        "",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # test = datasets.MNIST(
    #     "",
    #     train=False,
    #     download=True,
    #     transform=transforms.Compose([transforms.ToTensor()]),
    # )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = AutoEncoder(inputs=784).to(device)

    # Optimiser, which im currently coding a manual version of it .

    optimiser = optim.Adam(net.parameters(), lr=1e-3)

    loss_function = nn.MSELoss()

    trainset = torch.utils.data.DataLoader(
        train, batch_size=512, shuffle=True, num_workers=4, pin_memory=True
    )
    # testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

    test_examples = None

    epoch = 1

    testset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=False)

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

    # This code is a god teir piece of code
    # That would allow one to see what is goign on .
    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for i in range(number):
            # display original
            plotter = plt.subplot(2, number, i + 1)
            plt.imshow(test_examples[i].cpu().numpy().reshape(28, 28))
            plotter.get_xaxis().set_visible(False)
            plotter.get_yaxis().set_visible(False)
            plt.gray()
            # display reconstruction
            plotter = plt.subplot(2, number, i + 1 + number)
            plt.imshow(recon[i].cpu().numpy().reshape(28, 28))
            plotter.get_xaxis().set_visible(False)
            plotter.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    hook()
    main()
