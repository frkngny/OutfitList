import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools


interpreter = tf.lite.Interpreter(model_path="fashion_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

import cv2 as cv
im2 = cv.imread('dress.png')
im2 = cv.resize(im2, (28, 28))
grayIm = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

grayIm = tf.cast(grayIm, tf.float32)
arr = np.array(grayIm)
arr = np.expand_dims(arr, axis=0)
arr = np.expand_dims(arr, axis=3)

# Test model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(arr, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
