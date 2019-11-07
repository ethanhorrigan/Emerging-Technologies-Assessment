from flask import Flask, escape, request

# Creating an instance of Flask
# This is needed so that Flask knows where to look for templates, static files, and so on.
app = Flask(__name__)


# Using the route() decorator to tell Flask what URL should trigger a function.
@app.route("/")
def home():
        return app.send_static_file('index.html')

@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'