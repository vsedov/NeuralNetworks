#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-12-14 Viv Sedov
#
# File Name: test.py
# Distributed under terms of the MIT license.
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import sys

import torch.nn as nn
import torch.nn.functional as F
from frosch import hook

hook()


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 30), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(30, 784), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def is_venv():
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def main() -> None:
    model = Test().to(0)
    crit = nn.MSELoss()


if __name__ == "__main__":
    main()
