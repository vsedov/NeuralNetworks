#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-28 Viv Sedov
#
# File Name: PyTorchMinst.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import logging
from collections import Counter
from itertools import count

import matplotlib.pyplot as plt
import torch
from frosch import hook
from pprintpp import pprint as pp
from torchvision import datasets, transforms

logging.basicConfig(filename="Log_File.log", level=logging.INFO)


def torch_data_set():

    # Type of sample data .
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
        transform=transforms.Compose([transforms.ToTensor]),
    )

    trainset = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

    for data in trainset:
        pass
    x, y = data[0][0], data[1][0]

    plt.imshow(x.view([28, 28]))
    plt.show()

    lister = []
    for data in trainset:
        x, y = data  # data has 2 set
        lister.extend(int(i) for i in y)
    x = Counter(lister)
    for i in range(len(x)):
        pp(f"{i} :: {x[i] / len(lister) * 100}")


def main() -> None:

    torch_data_set()


if __name__ == "__main__":
    hook()
    main()
