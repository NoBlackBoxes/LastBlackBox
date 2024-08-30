from flask import Flask

# Create Flask app
app = Flask(__name__)

# Specify root ("/") route
@app.route("/")
def index():
    return "<h1>Hello ?!</h1>"

# Specify dynamic ("/<name>") route
@app.route("/<name>")
def user(name):
    Name = name.capitalize()
    return "<h1>Hello {0}!</h1>".format(Name)
    
#FIN