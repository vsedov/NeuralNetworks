#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-12-10 Viv Sedov
#
# File Name: test.py
# Distributed under terms of the MIT license.
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import better_exceptions
from frosch import hook

better_exceptions.hook()
hook()

import sys


def main():
    def is_venv():
        return hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

    if is_venv():
        print("inside virtualenv or venv")
    else:
        print("outside virtualenv or venv")


if __name__ == "__main__":
    main()
