# -*- coding: utf-8 -*-
"""
Build the LBB site

@author: kampff
"""

# Import libraries
import os
import Site.utilities as Utilities

# Import modules
import Site.course as Course
import Site.session as Session
import Site.box as Box
import Site.lesson as Lesson
import Site.instruction as Instruction
import Site.image as Image
import Site.video as Video
import Site.task as Task
import Site.input as Input
import Site.project as Project

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