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
        f"{pointer} bias  {pointer_bias} and once put into relu, you would get {relupointer}"
    )

    print(f"\n \n the neuron would tke this input . {pointer_bias}")

    # Relu
    # back propergation stage of this
    z = relu(pointer_bias)

    dvalue = 1
    drelu_dz = dvalue * (1.0 if z > 0 else 0)

    # -----------------------------#
    # Partial Derivative of teh chain
    # Rule : Relu * sumunation
    dsum_dxw0 = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0

    dsum_dxw1 = 1
    drelu_dxw1 = drelu_dz * dsum_dxw1

    dsum_dxw2 = 1
    drelu_dxw2 = drelu_dz * dsum_dxw2

    dsum_db = 1
    drelu_db = drelu_dz * dsum_db

    print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

    # ----------------------------#
    # Sumunation * x[i]w[i]

    dmul_dx0 = w[0]
    dmul_dx1 = w[1]
    dmul_dx2 = w[2]

    dmul_dw0 = x[0]
    dmul_dw1 = x[1]
    dmul_dw2 = x[2]

    drelu_dx0 = drelu_dxw0 * dmul_dx0
    drelu_dw0 = drelu_dxw0 * dmul_dw0

    drelu_dx1 = drelu_dxw1 * dmul_dx1
    drelu_dw1 = drelu_dxw1 * dmul_dw1

    drelu_dx2 = drelu_dxw2 * dmul_dx2
    drelu_dw2 = drelu_dxw2 * dmul_dw2

    pp([drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2])


def relu(inputs):
    return np.max(inputs, 0)


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
