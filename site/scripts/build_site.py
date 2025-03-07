# -*- coding: utf-8 -*-
"""
Build the LBB site

@author: kampff
"""

# Imports
import os
import LBB.config as Config
import LBB.utilities as Utilities
import LBB.Engine.course as Course

# Reload libraries and modules
import importlib
importlib.reload(Config)
importlib.reload(Utilities)
importlib.reload(Course)

# Set paths
templates_folder = Config.site_root + "/templates"
courses_templates_folder = templates_folder + "/courses"

# Load and render each course
for course_name in Config.course_names:
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