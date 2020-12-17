#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-12-17 Viv Sedov
#
# File Name: SideUnderstandingVector.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import better_exceptions
import numpy as np
from frosch import hook
from pprintpp import pprint as pp

better_exceptions.hook()
hook()


def main() -> None:
    vectors = np.dot([1, 2], [2, 3])

    pp(vectors)

    print("another change")


if __name__ == "__main__":
    main()
