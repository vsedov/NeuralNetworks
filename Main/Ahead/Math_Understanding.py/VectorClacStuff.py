#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2021-01-02 Viv Sedov
#
# File Name: VectorClacStuff.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

from dataclasses import dataclass, field

import numpy as np
from frosch import hook
from pprintpp import pprint as pp


@dataclass
class Vectr3d:
    x: int = field(repr=False)
    y: int
    z: int

    def __post_init__(self):
        self.x = self.y + self.z


u = Vectr3d(10, 20, 10)
print(u.__post_init__())


def forward():
    x = [1, -2, 3]
    w = [-3, -1, 2]

    bias = 1

    pointer = [x[i] * w[i] for i in range(len(x))]
    pointer_bias = sum(pointer) + bias
    # Interesting, i had not known that numpy doe sthis too

    relupointer = relu(pointer_bias)
    print(
        f"For values {pointer} we add a bias to given inputs to get {pointer_bias} and once put into relu, you would get {relupointer}"
    )

    print(f"\n \n the neuron would tke this input . {pointer_bias}")

    z = relu(pointer_bias)
    dvalue = 1
    # ^  This is the dirivative of the next layer
    drelu_dz = dvalue * (1.0 if z > 0 else 0)
    # ^ this is the relu dx

    # This is teh back prop stage of going back .
    dsum_dxw0 = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0
    print(f"Using the Chaing Rule and partial Dirivative you would get \n{drelu_dxw0}")


def relu(inputs):
    return max(inputs, 0.001)


def main() -> None:
    forward()


if __name__ == "__main__":
    hook()
    main()
    """
    Dataclasses very obsure subject than the other ones 
    data classes are something ive grown in python . 
    A data class is a class whose sole purpose is to hold data 
    the class will have variables than can be acced and written to but there is no extra logic to it .

    So what does this do exactly ? 
    You define things with name and type 
    while the function of our class is limited, the point of the data class 
    is to increase efficiency and reduce errors in your code . 
    its much better to pass around a vector3d than a int variable , so having this is better . 

    """
