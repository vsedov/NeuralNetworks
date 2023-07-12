#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-28 Viv Sedov
#
# File Name: MinstDataSetTest.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import logging
from collections import Counter
from itertools import count

import matplotlib.pyplot as plt
import torch
from frosch import hook
from pprintpp import pprint as pp
from torchvision import datasets, transforms

torch.cuda.set_device(0)


def torch_data_set():

    # Type of sample data .
    train = datasets.MNIST(
        "",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test = datasets.MNIST(
        "",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor]),
    )

    trainset = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

    # We use shuffle to generlise - everything we can do to give neural network
    # We can Nerual Network , hmm so those weights have to be within random .

    for data in trainset:
        pp(data)
        break
    x, y = data[0][0], data[1][0]
    plt.imshow(x.view([28, 28]))
    plt.show()
    print("\n")

    # i want to do this with the counter tool, im curious to see how that would work, do this after this video .

    # What we are doing here is checking the data types ballance - in the example above
    # total = 0
    # counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    # for data in trainset:
    #     Xs, ys = data
    #     for y in ys:
    #         counter_dict[int(y)] += 1
    #         total += 1
    # # print(counter_dict)
    # for i in counter_dict:
    #     print("{}:{}".format(i, (counter_dict[i] / total) * 100))

    print("\n")
    lister = []
    for data in trainset:
        x, y = data  # data has 2 set
        lister.extend(int(i) for i in y)
    x = Counter(lister)
    for i in range(len(x)):
        print(f"{i} :: {x[i] / len(lister) * 100}")


def main() -> None:

    torch_data_set()


if __name__ == "__main__":
    hook()
    main()
"""
Personal Notes :
For stuff like this i always have logging, Makes life easier

Torch Vision : Comes with a bunch of datasets - Collection of data that is used with vision .
--> Nice big thing that we bench mark against . Could be more

"""
