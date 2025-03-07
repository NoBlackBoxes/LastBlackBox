# -*- coding: utf-8 -*-
"""
LBB: Student Class

@author: kampff
"""

# Import libraries
import os
import glob
import json
import csv
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules
import Design.svg as SVG
import Site.config as Config
import Site.course as Course

# Student Class
class Student:
    def __init__(self, student_id=None):
        self.id = None                  # Student ID
        self.password_hash = None       # Student password hash
        self.name = None                # Student name
        self.nickname = None            # Student nickname
        self.email = None               # Student email
        self.progress = {}              # Student progress dictionary {box name:depth}
        self.current_course = None      # Student current course name
        self.course = None              # Student course
        self.authenticated = False      # Student authenticated boolean
        self.loaded = False             # Student loaded boolean
        if (student_id != None):
            self.load(student_id)       # Load from student ID
        return
    
    def is_active(self):
        return True

    def get_id(self):
        return self.id

    def is_authenticated(self):
        return self.authenticated

    def is_anonymous(self):
        return False

    # Load student object from JSON
    def load(self, student_id):
        student_folder = f"{Config.data_root}/students/{student_id}"
        student_path = f"{student_folder}/data.json"

        # Does student (data file) exist?
        if not os.path.isfile(student_path):
            return False

        # Load student data
        with open(student_path, 'r') as file:
            dictionary = json.load(file)
            self.id = dictionary.get("id")
            self.password_hash = dictionary.get("password_hash")
            self.name = dictionary.get("name")
            self.nickname = dictionary.get("nickname")
            self.email = dictionary.get("email")
            self.progress = dictionary.get("progress")
            self.current_course = dictionary.get("current_course")        
        if self.current_course:
            student_course_slug = Course.get_slug_from_name(self.current_course)
            student_course_path = f"{student_folder}/courses/{student_course_slug}.json"
            self.course = Course.Course(path=student_course_path)
        self.loaded = True
        return True

    # Store student object in JSON file
    def store(self):
        student_folder = f"{Config.data_root}/students/{self.id}"
        student_path = f"{student_folder}/data.json"
        dictionary = {
            "id": self.id,
            "password_hash": self.password_hash,
            "name": self.name,
            "nickname": self.nickname,
            "email": self.email,
            "progress": self.progress,
            "current_course": self.current_course
        }
        with open(student_path, "w") as file:
            json.dump(dictionary, file, indent=4)
        return

    # Find student from email
    def find(self, student_email):
        students_folder = f"{Config.data_root}/students"
        student_folders = glob.glob(students_folder+"/*/")
        for student_folder in student_folders:
            student_id = student_folder.split('/')[-2]
            self.load(student_id)
            if (self.email == student_email):
                return self
        self.loaded = False
        return None

    # Authenticate student
    def authenticate(self, student_password):
        a = check_password_hash(self.password_hash, student_password)
        if check_password_hash(self.password_hash, student_password):
            self.authenticated = True
            return True
        return False

    # Summarize student progress    
    def summarize_progress(self):
        num_open = 0
        num_01 = 0
        num_10 = 0
        num_11 = 0
        for status in self.progress.values():
            if status != '00':
                num_open += 1
            if status == '01':
                num_01 += 1
            if status == '10':
                num_10 += 1
            if status == '11':
                num_11 += 1
        return [num_open, num_01, num_10, num_11]

    # Generate student badge (SVG)
    def generate_badge(self):
        self.generate_badge_parameters()
        student_badge_folder = f"{Config.data_root}/students/{self.id}/badge"
        box_parameters_path = f"{student_badge_folder}/box_parameters_badge.csv"
        animation_parameters_path = f"{student_badge_folder}/animation_parameters_badge.csv"
        svg = SVG.SVG(f"brain_badge", None, 100, 64, "0 0 100 64", with_profile=False, with_title=False, with_labels=True)
        svg_path = f"{student_badge_folder}/{self.id}_badge.svg"
        svg.animate(box_parameters_path, animation_parameters_path, True, False, True, svg_path)
        return

    # Generate student badge parameters
    def generate_badge_parameters(self):
        student_badge_folder = Config.data_root + f"/students/{self.id}/badge"
        box_parameters_template_path = f"{Config.static_root}/resources/box_parameters_badge.csv"
        box_parameters_student_path = f"{student_badge_folder}/box_parameters_badge.csv"
        box_parameters = np.genfromtxt(box_parameters_template_path, delimiter=",", dtype=str, comments='##')
        animation_parameters_template_path = f"{Config.static_root}/resources/animation_parameters_badge.csv"
        animation_parameters_student_path = f"{student_badge_folder}/animation_parameters_badge.csv"
        animation_parameters = np.genfromtxt(animation_parameters_template_path, delimiter=",", dtype=str, comments='##')
        for i, level in enumerate(self.progress.values()):
            if level == '00':
                fill = "#000000"
                stroke = "#FFFFFF"
            elif level == '01':
                fill = "#999999"
                stroke = "#FFFFFF"
            elif level == '10':
                fill = "#FFFFFF"
                stroke = "#AAAAAA"
            elif level == '11':
                fill = "#FFFF00"
                stroke = "#FFFFFF"
            box_parameters[i, 6] = fill
            box_parameters[i, 7] = stroke
        with open(box_parameters_student_path, 'w') as f:
            csv.writer(f).writerows(box_parameters)
        with open(animation_parameters_student_path, 'w') as f:
            csv.writer(f).writerows(animation_parameters)
        return

    # Load student badge
    def load_badge(self):
        student_badge_folder = f"{Config.data_root}/students/{self.id}/badge"
        student_badge_path = student_badge_folder + f"/{self.id}_badge.svg"
        with open(student_badge_path, 'r') as file:
            next(file)  # Skip first line
            svg_string = file.read()
        return svg_string

    # Download student badge
    def download_badge(self, folder):
        student_folder = f"{Config.data_root}/students/{self.id}"
        svg_path = f"{folder}/badge_{self.id}.svg"
        # Convert to PDF? blah blah
        return

# FIN