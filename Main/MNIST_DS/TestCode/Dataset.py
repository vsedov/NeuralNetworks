#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-28 Viv Sedov
#
# File Name: Dataset.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"


import numpy as np
import torch
import torchvision
from frosch import hook
from NeuralNet import Net
from pprintpp import pprint as pp
from torchvision import datasets, transforms


def data_set():
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


def main() -> None:
    data_set()


if __name__ == "__main__":
    hook()
    main()
