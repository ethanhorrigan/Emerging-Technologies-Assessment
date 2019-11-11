from flask import Flask, json, jsonify, render_template, request
from model import prediction

# imports for array-handling and plotting

# https://www.palletsprojects.com/p/flask/
# Creating an instance of Flask 
# This is needed so that Flask knows where to look for templates, static files, and so on.
app = Flask(__name__)

# Using the route() decorator to bind a function to a URL.
@app.route("/")
def home():
    return app.send_static_file('index.html')

