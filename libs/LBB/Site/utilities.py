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
        task_status.update({task_name : 1})
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

# Find and convert all markdown emphasis tags
def convert_emphasis_tags(text):
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    return text

# Find and convert all markdown links in the format [text](url)
def convert_markdown_links(text):
    return re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, text)

# Replace matched links (and convert internal repo links)
def replace_link(match):
    link_text, link_url = match.groups()
    if not link_url.startswith("http"):
        link_url = re.sub(r'^(?:\.\./)+', '', link_url)
        link_url = f"https://github.com/NoBlackBoxes/LastBlackBox/tree/master/{link_url}"
    return f'<a id="link" href="{link_url}" target="_blank">{link_text}</a>'

#FIN