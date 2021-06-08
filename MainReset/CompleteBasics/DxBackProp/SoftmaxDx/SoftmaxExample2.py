#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: SoftmaxExample2
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import numpy as np
import pyinspect as pi


def softmax(input_data: np.ndarray) -> np.ndarray:
    ex = np.exp(input_data)
    return ex / np.sum(ex)


def sm_dir(s: np.ndarray) -> np.ndarray:
    s_vector = s.reshape(s.shape[0], 1)
    print("\nVector\n")
    print(s_vector)
    s_matrix = np.tile(s_vector, s.shape[0])
    print("\nMatrix\n")
    print(s_matrix)

    return np.diag(s) - (s_matrix * np.transpose(s_matrix))


def main() -> None:
    # input vector
    x = np.array([0.1, 0.5, 0.4])

    # using some hard coded values for the weights
    # rather than random numbers to illustrate how
    # it works

    # flake8: noqa: N806
    W = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 0.1],
            [0.11, 0.12, 0.13, 0.14, 0.15],
        ]
    )
    z = np.dot(np.transpose(W), x)

    h = softmax(z)

    print(h)

    DS = sm_dir(h)
    print("\n\n")
    print(DS)


if __name__ == "__main__":
    pi.install_traceback()
    main()
