#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2021-01-07 Viv Sedov
#
# File Name: AutoEncoder_test.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import logging

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from frosch import hook
from torchvision import datasets, transforms
from tqdm import tqdm

logging.basicConfig(filename="AutoEncoder.log", level=logging.DEBUG)

torch.device("cuda:0")


class AE(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputs, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
        )
        # Make a conv2d layer, just for the sake of it .
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, inputs),
            nn.LeakyReLU(),
        )

    def forward(self, inputs):
        encode = self.encoder(inputs)
        decode = self.decoder(encode)
        return decode


def main() -> None:
    device = torch.cuda.is_available()
    print(device)

    device = torch.device("cuda:0")
    print("On Gpu")

    # Increase Guasian Blur
    transform = transforms.Compose(
        [
            transforms.GaussianBlur(11, sigma=(0.1, 2.7)),
            transforms.ToTensor(),
        ]
    )

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
        transform=transform,
    )

    net = AE(inputs=784)
    net = net.cuda()

    print(torch.cuda.is_initialized())
    # Optimiser, which im currently coding a manual version of it .

    print(torch.cuda.current_device())

    optimiser = optim.Adam(net.parameters(), lr=1e-3)

    loss_function = nn.MSELoss()

    trainset = torch.utils.data.DataLoader(
        train, batch_size=512, shuffle=True, num_workers=4, pin_memory=True
    )
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

    test_examples = None

    epoch = 50

    for epoch in range(epoch):

        loss_count = 0
        for data, _ in tqdm(trainset):

            data = data.cuda()
            data = data.view(-1, 784)

            optimiser.zero_grad()  # What this should be linked to ?
            output = net(data)
            output.cuda()

            train_loss = loss_function(output, data)
            train_loss.backward()
            optimiser.step()
            loss_count += train_loss.item()
        print(
            f"epoch {epoch +1 } / {100} loss = {loss_count} and given loss is {train_loss}"
        )

    test_examples = None

    with torch.no_grad():
        for data in testset:
            data = data[0]
            data = data.cuda()
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
            # display reconstruction
            plotter = plt.subplot(2, number, i + 1 + number)
            plt.imshow(recon[i].cpu().numpy().reshape(28, 28))
            plotter.get_xaxis().set_visible(False)
            plotter.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    hook()
    main()
