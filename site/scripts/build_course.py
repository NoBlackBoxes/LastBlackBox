# -*- coding: utf-8 -*-
"""
Build the LBB site

@author: kampff
"""

# Import libraries
import os

# Import modules
import LBB.config as Config
import LBB.course as Course

# Reload libraries and modules
import importlib
importlib.reload(Config)
importlib.reload(Course)

# List all *available* courses
#course_names = ["Bootcamp", "Build a Brain"]
course_names = ["Build a Brain"]

# Load and render each course
for course_name in course_names:
    # Build course
    course = Course.Course(name=course_name)

    # Store course
    course.store(Config.data_root + "/test_course.json")

    # Load as new course
    new_course = Course.Course(path=Config.data_root + "/test_course.json")

    # Store new course
    new_course.store(Config.data_root + "/test_new_course.json")

# FIN