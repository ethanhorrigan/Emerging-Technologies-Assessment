# Author: Ethan Horrigan
# Description: This program runs the flask app to allow for hand written digits to be saved as images and
# predicited using the previous trained model

from flask import Flask, escape, request, render_template
import keras as kr
import numpy as np
import re
import tensorflow as tf
import base64
from keras.models import load_model
from keras import backend as K
import imageio
import cv2
from PIL import Image
from io import BytesIO

# https://www.palletsprojects.com/p/flask/
# Creating an instance of Flask 
# This is needed so that Flask knows where to look for templates, static files, and so on.
app = Flask(__name__)

def getModel():
    model = load_model('model.h5')
    return model

# Using the route() decorator to bind a function to a URL.
# adapted from: https://flask.palletsprojects.com/en/1.1.x/quickstart/#rendering-templates
@app.route('/')
def canvas():
    return render_template('index.html')

@app.route('/predict' , methods=['POST'])
def predict():
    # Getting the Image64 from from the image post
    imageB64 = request.values.get("imageBase64", "")
    
    # Decode the Image
    decode = base64.b64decode(imageB64[22:])
    
    # Open the Original Image
    with open("image.png", "wb") as f:
        f.write(decode)
    
    # Defining the width and height for the image resize
    width = 28
    height = 28
    # Dimensions = Width x Height
    dim = (width, height)
    
    # Open the Image
    img = Image.open("image.png")
    # img = Image.open("image.png").convert('L')
    # Save the Original Image (For Comparison)

    # Save the Original Image
    img.save("OriginalImage.png")

    # Downsize the Image to 28 x 28
    # https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    img = img.resize(dim, Image.ANTIALIAS)
    # Save the Resized Image
    img.save("ResizedImage.png")
    
    # Old Resize Method, did not work as intended so I opted for the PIL libray instead.
    # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    # img = cv2.resize(cv2.UMat(img), dsize=(28, 28), interpolation=cv2.INTER_NEAREST)

    # https://www.geeksforgeeks.org/python-pil-image-point-method/
    # http://effbot.org/imagingbook/introduction.htm
    # img point is used to try and preserve the edges and lines for more accurate predictions
    
    # Define the Threshold (0 - 255)
    threshold = 0
    # img = img.point(lambda p: p > threshold and 255)
    pt = lambda p: p > threshold and 255
    # Convert the Image to Black & White and Create points to preserve lines
    img = img.convert('L').point(pt)
    # Save the image in bytes.
    img.save("image.png")

    # Flatten (make one dimensional) and reshape the array without changing it's data.
    # Convert the data to float or int so we can divide it by 255
    # Dividing by 255 will give us either a 1 or a 0.
    # 1 represents a drawn pixel.
    # 0 represents a pixel that has not been drawn on.
    # https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/
    # grayArray = np.ndarray.ravel(np.array(gray)).reshape(1, 784).astype("float32") / 255
    flatArray = np.ndarray.ravel(np.array(img)).reshape(1, 784).astype("uint8") / 255
    #grayArray = ~np.ravel(gray).reshape(1, 784).astype(np.uint8) / 255.0

    # print("Printing image to array")
    # print(flatArray)
    # Predict the image.

    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
    # To clear the current session
    K.clear_session()
    # Retrieve the Trained MOdel
    model = getModel()
    # Predict with the given model
    predict = model.predict(flatArray)
    # Retrieve the prediction as a string
    prediction = str(np.argmax(predict))

    return prediction

# Clear the session to allow for prediction again.
K.clear_session()

# https://stackoverflow.com/questions/58015489/flask-and-keras-model-error-thread-local-object-has-no-attribute-value
# Run the app.
if __name__ == "__main__":
    app.run(debug=False, threaded=False)