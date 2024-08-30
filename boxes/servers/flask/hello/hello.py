from flask import Flask

# Create Flask app
app = Flask(__name__)

# Specify root ("/") route
@app.route("/")
def index():
    return "<h1>Hello Internet!</h1>"

#FIN