# -*- coding: utf-8 -*-
"""
LBB Web Application

@author: kampff
"""
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')

# Set library paths
import sys
sys.path.append(libs_path)
#----------------------------------------------------------

import os
from flask import Flask, render_template, send_file, request, redirect, session
from flask_login import LoginManager, current_user, login_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules
import LBB.user as User

# Define constants
UPLOAD_FOLDER = base_path + '/_tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create App
app = Flask(__name__)

# Config App
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

# Create login
login_manager = LoginManager(app)

@login_manager.user_loader
def load_user(user_id):
    user = User.User()
    user.load(user_id)
    if user.loaded:
        return user
    else:
        return None

###############################################################################
# Helper Functions
###############################################################################
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

###############################################################################
# Routes
###############################################################################

# Serve manifest (for PWA)
@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')

# Serve service worker
@app.route('/service_worker.js')
def serve_sw():
    return send_file('service_worker.js', mimetype='application/javascript')

## Serve home
#@app.route('/', methods=['GET', 'POST'])
#def upload_file():
#    if request.method == 'POST':
#        # check if the post request has the file part
#        if 'file' not in request.files:
#            flash('No file part')
#            return redirect(request.url)
#        file = request.files['file']
#        # if user does not select file, browser also
#        # submit an empty part without filename
#        if file.filename == '':
#            flash('No selected file')
#            return redirect(request.url)
#        if file and allowed_file(file.filename):
#            filename = secure_filename(file.filename)
#            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#            return render_template('index.html')
#    return render_template('index.html')

# Serve Home
@app.route('/')
def homepage():
    return render_template('index.html')

# Serve Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect('user')
    if request.method == 'POST':
        user_name = request.form['user_name']
        user_password = request.form['user_password']

        # Validate form input
        if (user_name == '') or (user_password == ''):
            return render_template('login.html', error="Please enter a valid LBB ID and password.")

        # Retrieve user
        user = User.User()
        user.load(user_name)
        if not user.loaded:
            return render_template('login.html', error="LBB ID not found. Have you registered?")

        # Validate password
        if check_password_hash(user.password_hash, user_password):
            login_user(user)
            return redirect('user')
        else:
            return render_template('login.html', error="Incorrect password.")

    return render_template('login.html')

# Serve User
@app.route('/user')
def user():
    return render_template('user.html')

# Serve Instructor
@app.route('/instructor')
def instructor():
    return render_template('instructor.html')

# Serve Topic
@app.route('/<box>/<topic>', methods=['GET', 'POST'])
def topic(box, topic):
    print(box, topic)
    if request.method == 'GET':
        route_url = f"{box}/{topic}.html"
        print(route_url)
    elif request.method == 'POST':
        form = request.form
        print(form.keys())
        for key in form.keys():
            for value in form.getlist(key):
                print(key,":",value)                
        ## Check if the post request has the file part
        #if 'file' not in request.files:
        #    flash('No file part')
        #    return redirect(request.url)
        file = request.files['file']
        ## if user does not select file, browser also
        ## submit an empty part without filename
        #if file.filename == '':
        #    flash('No selected file')
        #    return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            route_url = f"{box}/{topic}.html"
    else:
        route_url = f"{box}/{topic}.html"
    return render_template(route_url)

###############################################################################
# Run Application
###############################################################################
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

#FIN