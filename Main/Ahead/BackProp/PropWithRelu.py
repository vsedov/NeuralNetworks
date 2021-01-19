#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2021-01-18 Viv Sedov
#
# File Name: PropWithRelu.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import numpy as np
from frosch import hook
from pprintpp import pprint as pp


def main() -> None:
    # Layer infomation
    z = np.array([[1, 2, -3, -4], [2, -7, -1, 3], [-1, 2, 5, -1]])
    dvalues = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Assuming that this is the given batch data of those values
    drelu = dvalues.copy()
    drelu[z <= 0] = 0
    print(drelu)

    # drelu = np.zeros_like(z)
    # drelu[z > 0] = 1  # Inside control, infomation
    # pp(drelu)

    # pp(dvalues)
    # drelu *= dvalues

    ## These woudl be the final values that would get pushed forward
    # print("\nFinal Values:")
    # pp(drelu)

    ##: Sheet two here :


if __name__ == "__main__":
    hook()
    main()
