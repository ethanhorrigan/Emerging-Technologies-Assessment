from flask import Flask, json, jsonify, render_template, request
import keras as kr
from keras.models import load_model
import gzip
from model import prediction

# Creating an instance of Flask
# This is needed so that Flask knows where to look for templates, static files, and so on.
app = Flask(__name__)

# Using the route() decorator to bind a function to a URL.
@app.route("/")
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
if __name__ == "__main__":
    app.run(debug = True, threaded = False)