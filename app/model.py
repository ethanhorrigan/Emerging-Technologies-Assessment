# imports for array-handling and plotting
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


# load the MNIST dataset and split into (train & test sets) (Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.)
# returns: 
# x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
# y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
# ADAPTED FROM: https://keras.io/datasets/
def loadDataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
    print(x_train.shape)
    print(x_test.shape)
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
    
# Building linear stack of layears with sequential model
def buildNetwork():

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
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model

def predictImage(img):
    #Try/Except from https://www.w3schools.com/python/python_try_except.asp
    try:
        model = load_model("model.h5")
    except:
        print("Loading Failed..")
        print("Building Network..")
        model = buildNetwork()
        iPredict = model.predict(img)

        return iPredict.argmax()