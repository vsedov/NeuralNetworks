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


class Net(nn.Module):
    def __init__(self):
        super().__init__()  # Kernel Size - or a 5 x 5 kernel / window
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # This is some fake data that you parse through to get the values
        # This is somethign that oyu can get rid over time as well, if you would wish
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        # -1 = what ever you want .
        # 1 x 50 x  50 - this is how ever many feature set that you have
        # or the given image size

        self.to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self.to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    # Fake forward parse , this is not the whole forward method, but merely a part of it .
    # This only does teh first three layer forward .
    def convs(self, x):

        # Check the shape, and then multiply the dimension
        # So what we want to get that given data, adn those varialbes .
        # self.convs => Part of the forward method .

        # x = F.max_pool2d(F.relu(self.conv(X)) (2,2))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        print(x[0].shape)
        # this with this is that you dont really know what everything is properly
        # That is you dont know the gievn shape that you have ,. so you would have to deal with that

        print(x[0].shape[0] * x[0].shape[1] * x[0].shape[2])

        if self.to_linear is None:
            self.to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)  # all conv layer
        x = x.view(-1, self.to_linear)  # flattern
        x = F.relu(self.fc1(x))  # First fc1
        x = F.relu(self.fc2(x))

        return x  # No need to do softmax one value that oyu have right .


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

    dogs_vs_cats = DogsVsCats()
    training_data = np.load("training_data.npy", allow_pickle=True)
    # print(len(training_data[1]))

    # cv2.imshow("name", cv2.resize(training_data[1][0], (500, 500)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for data in training_data:
        pp(data[:5])
        # So this would give some sort of given test set for all given data

    pointer = Net()


if __name__ == "__main__":
    hook()
    main()
