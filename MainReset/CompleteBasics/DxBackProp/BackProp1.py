#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: BackProp1
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"


def main() -> None:
    x = [1.0, -2.0, 3.0]
    w = [-3.0, -1.0, 2.0]
    b = 1.0

    # Multiplying inputs by weights
    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]

    # Adding the bias
    z = xw0 + xw1 + xw2 + b
    # This becomes the next ouput

    # We have to do relu to the functional
    y = max(z, 0)
    print("before y before back prop -> z =  ", y)

    # ----------------------------------------------------------------
    # Doing the backwards pass  Single neuron back prop

    # Derivative from the next layer
    dvalue = 1.0

    # relu using the chain rule, in this case we state that the next layer dx
    # would be zero
    drelu_dz = dvalue * (1.0 if z > 0 else 0.0)

    print("Chain rule on rleu ", drelu_dz)

    # partial derivative of the multiplication using chain rule
    # ∂f/∂xₙ∙wₙ [(∑ (xw+b))]  === 1
    dsum_dxw0 = 1
    dsum_dxw1 = 1
    dsum_dxw2 = 1
    dsum_db = 1

    drelu_dxw0 = drelu_dz * dsum_dxw0
    drelu_dxw1 = drelu_dz * dsum_dxw1
    drelu_dxw2 = drelu_dz * dsum_dxw2
    drelu_db = drelu_dz * dsum_db

    print(
        "Sumunation rule applied partial dx on x0w0: ",
        drelu_dxw0,
        drelu_dxw1,
        drelu_dxw2,
        drelu_db,
        sep="\n",
    )

    # ∂f/ ∂x₀[x₀∙w₀]=w₀

    # Keep in mind this would have to be in batches as well
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

    print("input : ", [drelu_dx0, drelu_dx1, drelu_dx2])

    print("weights : ", [drelu_dw0, drelu_dw1, drelu_dw2])

    gradient_inputs = [drelu_dx0, drelu_dx1, drelu_dx2]
    gradient_weights = [drelu_dw0, drelu_dw1, drelu_dw2]
    gradient_bias = 1

    print("\n")

    print(
        "Grad Inputs : ",
        gradient_inputs,
        "Grad Weights : ",
        gradient_weights,
        "Grad Bias : ",
        gradient_bias,
        sep="\n",
    )

    # Applying the back prop on teh relu function

    w[0] += -0.001 * gradient_weights[0]
    w[1] += -0.001 * gradient_weights[1]
    w[2] += -0.001 * gradient_weights[2]

    b += -0.001 * gradient_bias

    print(w, b)

    xw0 = x[0] * w[0]
    xw1 = x[1] * w[1]
    xw2 = x[2] * w[2]

    z = xw0 + xw1 + xw2 + b

    # Neuron input is now lower using this chained method , where you have
    # a gradient of data that you would have to pass down
    print(max(z, 0))


if __name__ == "__main__":
    main()
