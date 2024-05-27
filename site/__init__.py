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

# Import libraries
import os
from flask import Flask, render_template, send_file, request, redirect, session
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules
import LBB.user as User
import LBB.utilities as Utilities

# Define constants
UPLOAD_FOLDER = base_path + '/_tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create App
app = Flask(__name__)

# Config App
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
app.config['MAIL_SERVER'] = 'smtp.protonmail.ch'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.getenv('PROTONMAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('PROTONMAIL_SMTP_TOKEN')
app.config['MAIL_DEFAULT_SENDER'] = 'info@voight-kampff.tech'

# Create mail sender
mail = Mail(app)

# Create login manager
login_manager = LoginManager(app)
@login_manager.user_loader
def load_user(user_id):
    user = User.User(user_id)
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

# Serve service worker (for PWA)
@app.route('/service_worker.js')
def serve_sw():
    return send_file('service_worker.js', mimetype='application/javascript')

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
        user_id = request.form['user_id']
        user_password = request.form['user_password']

        # Validate form input
        if (user_id == '') or (user_password == ''):
            return render_template('login.html', error="Please enter a valid LBB user ID and password.")

        # Retrieve user
        user = User.User(user_id)
        if not user.loaded:
            return render_template('login.html', error="LBB user ID not found. Have you registered?")

        # Validate password
        if check_password_hash(user.password_hash, user_password):
            user.authenticated = True
            login_user(user)
            return redirect('user')
        else:
            return render_template('login.html', error="Incorrect password.")

    return render_template('login.html')

# Serve Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('login')

# Serve User
@app.route('/user')
@login_required
def user():
    print(current_user.name)
    return render_template('user.html', user=current_user)

# Serve Reset
@app.route('/reset')
def reset():
    msg = Message(
            'Hello',
            recipients=['adam.kampff@gmail.com'],
            body='This is a test email sent from Flask-Mail!'
        )
    mail.send(msg)
    return redirect('login')

# Serve Instructor
@app.route('/instructor')
@login_required
def instructor():
    return render_template('instructor.html')

# Serve Topic
@app.route('/<box>/<topic>', methods=['GET', 'POST'])
@login_required
def topic(box, topic):
    print(box, topic)
    task_status = None
    if request.method == 'GET':
        route_url = f"{box}/{topic}.html"
        print(route_url)
    elif request.method == 'POST':
        form = request.form
        topic_folder_path = f"{app.config['UPLOAD_FOLDER']}/users/{current_user.id}/{box}/{topic}"
        Utilities.create_folder(topic_folder_path)
        task_name = request.form['task_name']
        # Retrieve Task Status (0=incomplete, 1=complete)
        # - Check user folder for submission for this task
        for key in form.keys():
            for value in form.getlist(key):
                print(key,":",value)
        task_file_path = f"{topic_folder_path}/{task_name}.txt"
        f = open(task_file_path, 'w')
        f.write("yay")
        f.close()
        route_url = f"{box}/{topic}.html"
        task_status = {task_name : 1}
        print(task_status)
        ## Validate form submission
        #print(form.keys())
        #for key in form.keys():
        #    for value in form.getlist(key):
        #        print(key,":",value)
        ## Is there a file to upload?
        #if 'file' not in request.files:
        #    file = request.files['file']
        #    ## if user does not select file, browser also
        #    ## submit an empty part without filename
        #    if file.filename == '':
        #        flash('No selected file')
        #        return redirect(request.url)
        #    if file and allowed_file(file.filename):
        #        filename = secure_filename(file.filename)
        #        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #        route_url = f"{box}/{topic}.html"
    else:
        route_url = f"{box}/{topic}.html"
    return render_template(route_url, task_status=task_status)

###############################################################################
# Run Application
###############################################################################
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

#FIN