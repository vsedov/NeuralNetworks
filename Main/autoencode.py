import cv2
import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential, load_model

model = load_model("C:\\Users\\Rohith\\Documents\\Rohith_Stuff\\Datasets\\auto_en.h5")

test = x_train[1].reshape(1, 784)
y_test = model.predict(test)

inp_img = []
temp = []
for i in range(len(test[0])):
    temp.append(test[0][i])
    if (i + 1) % 28 == 0:
        inp_img.append(temp)
        temp = []
out_img = []
temp = []
for i in range(len(y_test[0])):
    temp.append(y_test[0][i])
    if (i + 1) % 28 == 0:
        out_img.append(temp)
        temp = []
inp_img = np.array(inp_img)
out_img = np.array(out_img)

cv2.imshow("Test Image", inp_img)
cv2.imshow("Output Image", out_img)
cv2.waitKey(0)
