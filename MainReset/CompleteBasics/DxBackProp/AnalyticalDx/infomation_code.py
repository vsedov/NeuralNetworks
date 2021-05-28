#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © Viv Sedov
#
# File Name: infomation_code
__author__ = "Viv Sedov"
__email__ = "viv.sv@hotmail.com"

import nnfs
import pyinspect as pi


def main() -> int:
    print(list(map(lambda x: x * 10, range(1, 10))))
    # Another way of doing this coudl be with teh following

    print([i * 10 for i in range(1, 10)])


if __name__ == "__main__":
    pi.install_traceback()
    nnfs.init()
    main()
