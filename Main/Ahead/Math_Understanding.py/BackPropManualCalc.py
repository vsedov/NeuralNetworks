#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2021-01-02 Viv Sedov
#
# File Name: BackPropManualCalc.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

#: Whole Cell
from dataclasses import dataclass, field

import numpy as np
from frosch import hook
from pprintpp import pprint as pp


def relu(inputs):
    return np.max(inputs, 0)


def leaky_relu(inputs):
    return np.max(0.01, inputs)


def forward():
    x = [1, -2, 3]
    w = [-3, -1, 2]

    b = 1

    pointer = [x[i] * w[i] for i in range(len(x))]
    pointer_bias = sum(pointer) + b

    # Relu
    # back propergation stage of this
    z = relu(pointer_bias)

    dvalue = 1
    drelu_dz = dvalue * (1.0 if z > 0 else 0)
    print("Dx of Relu ", drelu_dz, "\n")

    # -----------------------------#
    # Partial Derivative of teh chain
    # Rule : Relu * sumunation
    dsum_dxw0 = 1
    dsum_dxw1 = 1
    dsum_dxw2 = 1
    dsum_db = 1

    drelu_dxw0 = drelu_dz * dsum_dxw0
    drelu_dxw1 = drelu_dz * dsum_dxw1
    drelu_dxw2 = drelu_dz * dsum_dxw2
    drelu_db = drelu_dz * dsum_db
    print("Relu Gradients : ", x := ([drelu_dxw0, drelu_dxw1, drelu_dxw2]))
    print("\n")
    # Sumunation * x[i]w[i]

    dmul_dx0 = w[0]
    dmul_dx1 = w[1]
    dmul_dx2 = w[2]

    dmul_dw0 = x[0]
    dmul_dw1 = x[1]
    dmul_dw2 = x[2]

    drelu_dx0 = drelu_dxw0 * dmul_dx0
    drelu_dx1 = drelu_dxw1 * dmul_dx1
    drelu_dx2 = drelu_dxw2 * dmul_dx2

    drelu_dw0 = drelu_dxw0 * dmul_dw0
    drelu_dw1 = drelu_dxw1 * dmul_dw1
    drelu_dw2 = drelu_dxw2 * dmul_dw2

    # Lets now say that we want to make a gradient out of those partial dx
    dw = [drelu_dw0, drelu_dw1, drelu_dw2]
    dx = [drelu_dx0, drelu_dx1, drelu_dx2]

    print("Original Inputs", x)
    print("Original Weights", w, "\n")
    # Apply Fractional pointer within w and dw
    print("Partial Dw and Dx -> div 100")
    print("Grad Dw ", [i * -0.001 for i in dw])
    print("Grad Dx ", [i * -0.001 for i in dx])

    w[0] += -0.001 * dw[0]
    w[1] += -0.001 * dw[1]
    w[2] += -0.001 * dw[2]
    # Small changes to that that will make the main difference between the back prop

    b += 0.001 * drelu_db
    print("\n After W * Dw with 0.001", w, b)

    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]

    print("\n", (xw0 + xw1 + xw2) + b)

    print(w)


def main() -> None:
    forward()


if __name__ == "__main__":
    hook()
    main()
