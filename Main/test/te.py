#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-12-17 Viv Sedov
#
# File Name: te.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import better_exceptions
from frosch import hook

better_exceptions.hook()
hook()


def main() -> None:

    for pointer in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        print(pointer)


if __name__ == "__main__":
    main()
