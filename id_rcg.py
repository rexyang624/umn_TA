#!/usr/bin/python

import cv2
import sys
from keras.models import load_model
import numpy as np


MODEL_FILENAME = "MNIST_model.hdf5"

model = load_model(MODEL_FILENAME)

id_sequence = ""

for i in range(1,8):
    digit = cv2.imread(str(sys.argv[1])+str(i)+".jpg", 0).astype('float')/255.0
    digit = np.expand_dims(digit, axis=2)
    digit = np.expand_dims(digit, axis=0)
    prediction = model.predict(digit)
    id_sequence += str(prediction.argmax(axis=1)[0])
print(id_sequence)
