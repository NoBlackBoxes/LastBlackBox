# -*- coding: utf-8 -*-
"""
Build the LBB site

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
import LBB.utilities as Utilities

# Import modules
import LBB.course as Course
import LBB.session as Session
import LBB.box as Box
import LBB.lesson as Lesson
import LBB.instruction as Instruction
import LBB.image as Image
import LBB.video as Video
import LBB.task as Task
import LBB.input as Input
import LBB.project as Project

# Reload libraries and modules
import importlib
importlib.reload(Utilities)
importlib.reload(Course)
importlib.reload(Session)
importlib.reload(Box)
importlib.reload(Instruction)
importlib.reload(Image)
importlib.reload(Video)
importlib.reload(Task)
importlib.reload(Input)
importlib.reload(Project)

#----------------------------------------------------------

# Root paths
repo_root = os.path.dirname(base_path)
course_root = repo_root + "/course"
templates_folder = base_path + "/templates"
courses_templates_folder = templates_folder + "/courses"

# List all *available* courses
#course_names = ["bootcamp", "buildabrain"]
course_names = ["buildabrain"]

# Load and render each course
for course_name in course_names:
    # Load Course
    course = Course.Course(course_name)

    # Render Course
    course_template_folder = courses_templates_folder + "/" + course_name
    Utilities.clear_folder(course_template_folder)
    course.render(course_template_folder)

# FIN