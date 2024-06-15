# -*- coding: utf-8 -*-
"""
LBB: User Class

@author: kampff
"""
#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')
data_path = base_path + '/_tmp'

# Set library paths
import sys
sys.path.append(libs_path)
#----------------------------------------------------------

# Import libraries
import pickle
import glob
import csv
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules
import Design.svg as SVG

# User Class
class User:
    def __init__(self, _user_id=None):
        self.id = None              # ID
        self.password_hash = None   # Password hash
        self.name = None            # Name
        self.email = None           # Email
        self.instructor = False     # Instructor boolean
        self.admin = False          # Administrator boolean
        self.authenticated = False  # Authenticated boolean
        self.loaded = False         # Loaded boolean
        self.boxes = {}             # User box status dictionary
        self.current_course = None  # Current course
        self.current_box = None     # Current box
        self.current_topic = None   # Current topic
        self.progress = [0,0,0,0]   # Progress
        if (_user_id != None):      # Load from User ID
            self.load(_user_id)
        return
    
    def is_active(self):
        return True

    def get_id(self):
        return self.id

    def is_authenticated(self):
        return self.authenticated

    def is_anonymous(self):
        return False

    def store(self):
        user_data = {
            "id" : self.id,
            "password_hash" : self.password_hash,
            "name" : self.name,
            "email" : self.email,
            "instructor" : self.instructor,
            "admin" : self.admin,
            "boxes" : self.boxes,
            "current_course" : self.current_course,
            "current_box" : self.current_box,
            "current_topic" : self.current_topic,
        }
        user_folder = f"{data_path}/users/{self.id}"
        user_path = f"{user_folder}/user_data.pkl"
        with open(user_path, 'wb') as fp:
            pickle.dump(user_data, fp)
        return

    def load(self, user_id):
        user_folder = f"{data_path}/users/{user_id}"
        user_path = f"{user_folder}/user_data.pkl"
        # Does user (data file) exist?
        if not os.path.isfile(user_path):
            return False
        # Load user data
        with open(user_path, 'rb') as pickle_file:
            user_data = pickle.load(pickle_file)
        self.id = user_data['id']
        self.password_hash = user_data['password_hash']
        self.name = user_data['name']
        self.email = user_data['email']
        self.instructor = user_data['instructor']
        self.admin = user_data['admin']
        self.boxes = user_data['boxes']
        self.current_course = user_data['current_course']
        self.current_box = user_data['current_box']
        self.current_topic = user_data['current_topic']
        self.loaded = True
        return True

    def find(self, user_email):
        users_folder = f"{data_path}/users"
        user_folders = glob.glob(users_folder+"/*/")
        for user_folder in user_folders:
            user_id = user_folder.split('/')[-2]
            self.load(user_id)
            if (self.email == user_email):
                return True
        self.loaded = False
        return False

    def authenticate(self, user_password):
        if check_password_hash(self.password_hash, user_password):
            self.authenticated = True
            return True
        return False
    
    def update_progress(self):
        num_open = 0
        num_01 = 0
        num_10 = 0
        num_11 = 0
        for status in self.boxes.values():
            if status != '00':
                num_open += 1
            if status == '01':
                num_01 += 1
            if status == '10':
                num_10 += 1
            if status == '11':
                num_11 += 1
        self.progress = [num_open, num_01, num_10, num_11]
        return

    def generate_badge(self):
        user_folder = f"{data_path}/users/{self.id}"
        box_parameters_path = f"{user_folder}/box_parameters_badge.csv"
        animation_parameters_path = f"{user_folder}/animation_parameters_badge.csv"
        svg = SVG.SVG(f"brain_badge", None, 100, 100, "0 0 100 100", with_profile=False, with_title=False, with_labels=True)
        svg_path = f"{user_folder}/badge_{self.id}.svg"
        svg.animate(box_parameters_path, animation_parameters_path, True, False, True, svg_path)
        with open(svg_path, 'r') as file:
            next(file)
            svg_string = file.read()
        return svg_string

    def generate_badge_parameters(self, static_folder):
        user_folder = f"{data_path}/users/{self.id}"
        resources_folder = f"{static_folder}/resources"
        box_parameters_template_path = f"{resources_folder}/box_parameters_badge.csv"
        box_parameters_user_path = f"{user_folder}/box_parameters_badge.csv"
        box_parameters = np.genfromtxt(box_parameters_template_path, delimiter=",", dtype=str, comments='##')
        animation_parameters_template_path = f"{resources_folder}/animation_parameters_badge.csv"
        animation_parameters_user_path = f"{user_folder}/animation_parameters_badge.csv"
        animation_parameters = np.genfromtxt(animation_parameters_template_path, delimiter=",", dtype=str, comments='##')
        for i, level in enumerate(self.boxes.values()):
            if level == '00':
                fill = "#000000"
                stroke = "#FFFFFF"
            elif level == '01':
                fill = "#888888"
                stroke = "#FFFFFF"
            elif level == '10':
                fill = "#FFFFFF"
                stroke = "#AAAAAA"
            elif level == '11':
                fill = "#FFFF00"
                stroke = "#FFFFFF"
            box_parameters[i, 6] = fill
            box_parameters[i, 7] = stroke
        with open(box_parameters_user_path, 'w') as f:
            csv.writer(f).writerows(box_parameters)
        with open(animation_parameters_user_path, 'w') as f:
            csv.writer(f).writerows(animation_parameters)
        return

    def download_badge(self, static_folder):
        user_folder = f"{data_path}/users/{self.id}"
        svg_path = f"{user_folder}/badge_{self.id}.svg"
        # Convert to PDF? blah blah
        return

# FIN