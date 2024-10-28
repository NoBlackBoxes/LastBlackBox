# -*- coding: utf-8 -*-
"""
LBB Web Application

@author: kampff
"""

# Import libraries
import os
from flask import Flask, render_template, send_file, request, redirect, session
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import LBB.utilities as Utilities

# Import modules
import LBB.config as Config
import LBB.student as Student

# Define constants
UPLOAD_FOLDER = Config.site_root + '/_tmp'

# Create application
app = Flask(__name__)

# Configure application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = Config.flask_secret_key
app.config['MAIL_SERVER'] = 'smtp.protonmail.ch'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = Config.protonmail_username
app.config['MAIL_PASSWORD'] = Config.protonmail_smtp_token
app.config['MAIL_DEFAULT_SENDER'] = Config.protonmail_username

# Create mail sender
mail = Mail(app)

# Create login manager
login_manager = LoginManager(app)
@login_manager.user_loader
def load_student(student_id):
    student = Student.Student(student_id)
    if student.loaded:
        return student
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
        return redirect('student')
    if request.method == 'POST':
        student_id = request.form['student_id']
        student_password = request.form['student_password']
        if (len(student_id) != 6):
            return render_template('login.html', error="Please enter a six digit LBB student ID")
        student = Student.Student(student_id)
        if not student.loaded:
            return render_template('login.html', error="LBB student ID not found")
        if student.authenticate(student_password):
            login_user(student)
            return redirect('student')
        else:
            return render_template('login.html', error="Incorrect password")
    return render_template('login.html')

# Serve Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('login')

# Serve Student
@app.route('/student')
@login_required
def student():
    progress_summary = current_user.summarize_progress()
    badge = current_user.load_badge()
    return render_template('student.html', student=current_user, badge=badge, progress_summary=progress_summary)

# Serve Update
@app.route('/update', methods=['GET', 'POST'])
@login_required
def update():
    if request.method == 'POST':
        student_name = request.form['student_name']
        student_nickname = request.form['student_nickname']
        student_email = request.form['student_email']
        student_password = request.form['student_password']
        student_password_confirmation = request.form['student_password_confirmation']
        if student_name != '':
            current_user.name = student_name
        if student_nickname != '':
            current_user.nickname = student_nickname
        if student_email != '':
            current_user.email = student_email
        if student_password != '':
            if student_password != student_password_confirmation:
                return render_template('update.html', student=current_user, badge=badge, error="Password mismatch")
            else:
                current_user.password_hash = generate_password_hash(student_password)
        current_user.store()
        return redirect('student')
    return render_template('update.html', student=current_user, badge=badge)

# Serve Recovery
@app.route('/recovery', methods=['GET', 'POST'])
def recovery():
    if request.method == 'POST':
        student_email = request.form['student_email']
        if not Utilities.is_valid_email(student_email):
            return render_template('recovery.html', error="Please enter a valid email address.")
        student = Student.Student()
        student = student.find(student_email)
        if student != None:
            student_password = secrets.token_urlsafe(11)
            student.password_hash = generate_password_hash(student_password)
            student.store()
            msg = Message('LBB Login Details', recipients=[student_email], body=f"Your LBB login recovery details:\n Student ID: {student.id}\n Temporary Password: {student_password}\n\nHave a nice day!\nLBB Team")
            mail.send(msg)
            return render_template('sent.html', message=f"Recovery details sent to {student_email}. Redirecting to login page."), {"Refresh": "5; url=login"}
        else:
            return render_template('recovery.html', error="This email was not registered by any LBB students!")
    return render_template('recovery.html')

# Serve Course Schedule
@app.route('/<course_slug>', methods=['GET'])
@login_required
def schedule(course_slug):
    #student_session_folder_path = f"{app.config['UPLOAD_FOLDER']}/students/{current_user.id}/{course}/{session}"
    if request.method == 'GET':
        route_url = "course.html"
    else:
        route_url = "course.html"
    return render_template(route_url, student=current_user, course=current_user.course)

# Serve Course Lesson
@app.route('/<course_slug>/<session_slug>/<box_slug>/<lesson_slug>', methods=['GET', 'POST'])
@login_required
def session(course_slug, session_slug, box_slug, lesson_slug):
    #student_session_folder_path = f"{app.config['UPLOAD_FOLDER']}/students/{current_user.id}/{course}/{session}"
    #task_status = Utilities.retrieve_task_status(student_session_folder_path)
    course = current_user.course
    session = next((session for session in course.sessions if session.slug == session_slug), None)
    box = next((box for box in session.boxes if box.slug == box_slug), None)
    lesson = next((lesson for lesson in box.lessons if lesson.slug == lesson_slug), None)
    if request.method == 'GET':
        route_url = "lesson.html"
#    elif request.method == 'POST':
#        form = request.form
#        task_name = request.form['task_name']
#        Utilities.archive_task_submission(student_session_folder_path, task_name)
#
#        # Store new submission(s)
#        for key in form.keys():
#            for value in form.getlist(key):
#                print(key,":",value)
#        task_file_path = f"{student_session_folder_path}/{task_name}.txt"
#        f = open(task_file_path, 'w')
#        f.write("yay")
#        f.close()
#        route_url = "lesson.html"
#        #task_status.update({task_name : 1})
#        #print(task_status)
#        ## Validate form submission
#        #print(form.keys())
#        #for key in form.keys():
#        #    for value in form.getlist(key):
#        #        print(key,":",value)
#        ## Is there a file to upload?
#        #if 'file' not in request.files:
#        #    file = request.files['file']
#        #    ## if student does not select file, browser also
#        #    ## submit an empty part without filename
#        #    if file.filename == '':
#        #        flash('No selected file')
#        #        return redirect(request.url)
#        #    if file and is_allowed_file(file.filename):
#        #        filename = secure_filename(file.filename)
#        #        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#        #        route_url = f"{box}/{topic}.html"
    else:
        route_url = "lesson.html"
    return render_template(route_url, student=current_user, course=course, session=session, box=box, lesson=lesson)

###############################################################################
# Run Application
###############################################################################
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

#FIN