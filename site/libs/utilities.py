# -*- coding: utf-8 -*-
"""
Utility Library

@author: kampff
"""

# Import libraries
import os
import glob
import shutil
import re

# Define constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Confirm folder (create if it does not exist)
def confirm_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return

# Clear a folder (or create)
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    return

# Permissable file format for task submission?
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Validate format for email address?
def is_valid_email(email_address):
    regex = r"[^@]+@[^@]+\.[^@]+"
    if re.fullmatch(regex, email_address):
        return True
    return False

# Retrieve task submission status for topic page
def retrieve_task_status(topic_folder_path):
    confirm_folder(topic_folder_path)
    task_status = {}
    submissions = glob.glob(topic_folder_path + f"/*.txt")
    for submission in submissions:
        task_name = os.path.basename(submission)[:-4]
        task_status = {task_name : 1}
    return task_status

# Archive previous task submissions for named task on topic page
def archive_task_submission(topic_folder_path, task_name):
    submissions = glob.glob(topic_folder_path + f"/*{task_name}*")
    num_submissions = len(submissions)
    if num_submissions > 0:
        confirm_folder(topic_folder_path + "/_archive")
        for submission in submissions:
            destination = topic_folder_path+f"/_archive/{os.path.basename(submission)}"
            shutil.move(submission, destination)
    return

#FIN