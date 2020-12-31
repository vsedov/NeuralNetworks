#! /usr/bin/env python3
# -- coding: utf-8 --
# vim:fenc=utf-8
#
# Copyright © 2020-12-30 Viv Sedov
#
# File Name: CatsVDogs.py
__author__ = "Viv Sedov"
__email__ = "viv.sb@hotmail.com"

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from frosch import hook
from pprintpp import pprint as pp
from tqdm import tqdm

torch.cuda.set_device(0)




class DogsVsCats(object):
    def __init__(self):
        self.IMG_SIZE = 50

        self.CATS = "kagglecatsanddogs_3367a/PetImages/Cat/"

        self.DOGS = "kagglecatsanddogs_3367a/PetImages/Dog/"

        # using one hot veector or 2 hot vecotr
        self.LABELS = {self.CATS: 0, self.DOGS: 1}
        # This allow us to get 1 hot vector and those are classes .

        self.training_data = []
        # Dogs and cats with label

        self.catCount = 0
        self.dogCount = 0

    # Counting of balance , and knowing how much data and difference there are  . make sure you have enough .

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)

                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                    # This part was more annyoing when i had a look at it at first
                    self.training_data.append(
                        [np.array(img), np.eye(2)[self.LABELS[label]]]
                    )
                    """
                    np.array(img)  to image form
                    then next to that we have np.eye(2) for 0 and 1
                    self.LABELS[label] - so in this case if its a dog or a cat .
                    that makes sense no ?
                    """

                    if label == self.CATS:
                        self.catCount += 1
                    elif label == self.DOGS:
                        self.dogCount += 1
                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: {}".format(self.catCount))

        print("Cats: {}".format(self.dogCount))


def main() -> None:
    REBUILD_DATA = False

    if REBUILD_DATA == True:
        dogs_vs_cats = DogsVsCats()
        dogs_vs_cats.make_training_data()

    # Though you can use this for matplot lib as well
    training_data = np.load("training_data.npy", allow_pickle=True)
    print(len(training_data[1]))

    # cv2.imshow("name", cv2.resize(training_data[1][0], (500, 500)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pointer = Net()

    plt.imshow(training_data[1][0], cmap="gray")

    plt.show()
    print(training_data[1][1])

    # This is just showing the given dataa that we have .


if __name__ == "__main__":
    hook()
    main()
