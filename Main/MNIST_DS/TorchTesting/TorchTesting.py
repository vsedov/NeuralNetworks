#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-28 Viv Sedov
#
# File Name: TorchTesting.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"


import torch
from frosch import hook


def torchStuff():
    x = torch.Tensor([5, 3])
    y = torch.Tensor([2, 1])
    print(x, y)
    x_1 = torch.zeros([2, 5])
    print(x_1)
    print(x_1.shape)

    y = torch.rand([2, 5])
    print(y)  # Random Variable for reshaping .
    # Reshaping a vecotr within numpy - np.reshape

    # Flattern  = 1,10
    y = y.view([1, 10])
    print(y)


def main() -> None:
    torchStuff()


if __name__ == "__main__":
    hook()
    main()
