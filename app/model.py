# Author: Ethan Horrigan
# Description: This program uses Convolutional Neural Networks (CNN) to classify handwritten digits as numbers 0 - 9

# imports for array-handling and plotting
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential, load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('agg')
# keras imports for the dataset and building our neural network

# Building linear stack of layears with sequential model

# load the MNIST dataset and split into (train & test sets) (Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.)
# returns:
# x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
# y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
# ADAPTED FROM: https://keras.io/datasets/
(X_train,Y_train), (X_test, Y_test) = mnist.load_data()
# reshape dataset to have a single channel
# Reshape the features ( X_train and X_test)to fit the model.
# train set will have 60,000 rows of 28 x 28 pixel 
# with depth=1 (gray scale).
# test set will have 10,000 rows of 28 x 28 pixel 
# with depth=1 (gray scale).

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one hot encode target values
# https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
n_classes = 10
print("Shape before one-hot encoding: ", Y_train.shape)
Y_train = np_utils.to_categorical(Y_train, n_classes)
Y_test = np_utils.to_categorical(Y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy',
                metrics=['accuracy'], optimizer='adam')

# Try/Except from https://www.w3schools.com/python/python_try_except.asp
try:
    model = load_model("model.h5")
except:
    print("Loading model failed")
    print("Building new Network..")
    # training the model and saving metrics in history
    history = model.fit(X_train, Y_train,batch_size=128, epochs=20,verbose=2,validation_data=(X_test, Y_test))
    model.save("model.h5")
    print('Saved trained model')
    # predictImg = model.predict(img)
