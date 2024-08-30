from flask import Flask, render_template

# Create Flask app
app = Flask(__name__)

# Specify root ("/") route
@app.route("/")
def index():
    return render_template("index.html")

# Specify dynamic ("/<name>") route
@app.route("/<name>")
def user(name):
    return render_template("name.html", name=name)
    
#FIN