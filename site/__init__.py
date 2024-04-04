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
from flask import Flask, render_template, send_file, request, redirect
from flask_login import LoginManager, current_user, login_user
from werkzeug.utils import secure_filename

# Import modules
import LBB.user as User

# Define constants
UPLOAD_FOLDER = base_path + '/_tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create App
app = Flask(__name__)

# Config App
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create login
login_manager = LoginManager(app)

@login_manager.user_loader
def load_user(user_id):
    # Load user (student or instructor) from filesystem
    user = User.get(user_id)
    return user

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
        return redirect('student.html')
    if request.method == 'POST':
        student_name = request.form['student_name']
        student_password = request.form['student_password']
        print(f"login!!!: {student_name}:{student_password}")
    return render_template('login.html')

# Serve Student
@app.route('/student')
def student():
    return render_template('student.html')

# Serve Instructor
@app.route('/instructor')
def instructor():
    return render_template('instructor.html')

# Serve Topic
@app.route('/<box>/<topic>')
def topic(box, topic):
    print(box, topic)
    if request.method == 'GET':
        route_url = f"{box}/{topic}.html"
        print(route_url)
    else:
        route_url = f"{box}/{topic}.html"
    return render_template(route_url)

###############################################################################
# Run Application
###############################################################################
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

#FIN