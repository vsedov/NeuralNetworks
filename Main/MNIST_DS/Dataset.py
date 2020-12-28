#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-28 Viv Sedov
#
# File Name: Dataset.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import logging

import numpy as np
import torch
import torchvision
from frosch import hook
from pprintpp import pprint as pp
from torchvision import datasets, transforms

logging.basicConfig(filename="LogForDataSet.log", level=logging.INFO)


def data_set():
    train = datasets.MNIST(
        "",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )


def main() -> None:

    pass


if __name__ == "__main__":
    hook()
    main()
