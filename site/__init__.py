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
from flask import Flask, render_template, send_file, request, redirect, session
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import LBB.utilities as Utilities

# Import modules
import LBB.user as User

# Define constants
UPLOAD_FOLDER = base_path + '/_tmp'

# Create application
app = Flask(__name__)

# Configure application
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
    return None

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

# Serve Root
@app.route('/')
def root():
    return redirect('login')

# Serve Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect('user')
    if request.method == 'POST':
        user_id = request.form['user_id']
        user_password = request.form['user_password']
        if (len(user_id) != 6):
            return render_template('login.html', error="Please enter a six digit LBB user ID")
        user = User.User(user_id)
        if not user.loaded:
            return render_template('login.html', error="LBB user ID not found")
        if user.authenticate(user_password):
            login_user(user)
            return redirect('user')
        else:
            return render_template('login.html', error="Incorrect password")
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
    current_user.update_progress()
    current_user.generate_badge_parameters(app.static_folder)
    badge = current_user.generate_badge()
    return render_template('user.html', user=current_user, badge=badge)

# Serve Update
@app.route('/update', methods=['GET', 'POST'])
@login_required
def update():
    current_user.update_progress()
    current_user.generate_badge_parameters(app.static_folder)
    badge = current_user.generate_badge()
    if request.method == 'POST':
        user_name = request.form['user_name']
        user_nickname = request.form['user_nickname']
        user_email = request.form['user_email']
        user_password = request.form['user_password']
        user_password_confirmation = request.form['user_password_confirmation']
        if user_name != '':
            current_user.name = user_name
        if user_nickname != '':
            current_user.nickname = user_nickname
        if user_email != '':
            current_user.email = user_email
        if user_password != '':
            if user_password != user_password_confirmation:
                return render_template('update.html', user=current_user, badge=badge, error="Password mismatch")
            else:
                current_user.password_hash = generate_password_hash(user_password)
        current_user.store()
        return redirect('user')
    return render_template('update.html', user=current_user, badge=badge)

# Serve Recovery
@app.route('/recovery', methods=['GET', 'POST'])
def recovery():
    if request.method == 'POST':
        user_email = request.form['user_email']
        if not Utilities.is_valid_email(user_email):
            return render_template('recovery.html', error="Please enter a valid email address.")
        user = User.User()
        user = user.find(user_email)
        if user != None:
            user_password = secrets.token_urlsafe(11)
            user.password_hash = generate_password_hash(user_password)
            user.store()
            msg = Message('LBB Login Details', recipients=[user_email], body=f"Your LBB login recovery details:\n User ID: {user.id}\n Temporary Password: {user_password}\n\nHave a nice day!\nLBB Team")
            mail.send(msg)
            return render_template('sent.html', message=f"Recovery details sent to {user_email}. Redirecting to login page."), {"Refresh": "5; url=login"}
        else:
            return render_template('recovery.html', error="This email was not registered by any LBB users!")
    return render_template('recovery.html')

# Serve Instructor
@app.route('/instructor')
@login_required
def instructor():
    return render_template('instructor.html')

# Serve Topic
@app.route('/<box>/<topic>', methods=['GET', 'POST'])
@login_required
def topic(box, topic):
    topic_folder_path = f"{app.config['UPLOAD_FOLDER']}/users/{current_user.id}/{box}/{topic}"
    task_status = Utilities.retrieve_task_status(topic_folder_path)
    if request.method == 'GET':
        route_url = f"{box}/{topic}.html"
    elif request.method == 'POST':
        form = request.form
        task_name = request.form['task_name']
        Utilities.archive_task_submission(topic_folder_path, task_name)

        # Store new submission(s)
        for key in form.keys():
            for value in form.getlist(key):
                print(key,":",value)
        task_file_path = f"{topic_folder_path}/{task_name}.txt"
        f = open(task_file_path, 'w')
        f.write("yay")
        f.close()
        route_url = f"{box}/{topic}.html"
        task_status.update({task_name : 1})
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
        #    if file and is_allowed_file(file.filename):
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