# -*- coding: utf-8 -*-
"""
Build the LBB site

@author: kampff
"""

# Import libraries
import os

# Import modules
import LBB.Site.config as Site_Config
import LBB.Site.utilities as Site_Utilities
import LBB.Engine.course as Course

# Reload libraries and modules
import importlib
importlib.reload(Site_Config)
importlib.reload(Site_Utilities)
importlib.reload(Course)

#----------------------------------------------------------

# Root paths
repo_root = os.path.dirname(base_path)
course_root = repo_root + "/course"
templates_folder = base_path + "/templates"
courses_templates_folder = templates_folder + "/courses"


# Course names
names = ["The Last Black Box", "Build a Brain", "Bootcamp"]

# Load and render each course
for name in names:
    # Set paths
    if name == "The Last Black Box":
        Config.image_prefix = "../.."
    else:
        Config.image_prefix = "../../../.."

    # Load course
    print(f"Loading \"{name}\"...")
    course = Course.Course(name)

    # Render HTML for each course
    course_template_folder = courses_templates_folder + "/" + course_name
    Utilities.clear_folder(course_template_folder)
    course.render(course_template_folder)

# FIN