# -*- coding: utf-8 -*-
"""
Generate LBB Courses

@author: kampff
"""

# Import Libraries
import os
import glob

# Import modules
import LBB.Engine.utilities as Utilities
import LBB.Engine.config as Config
import LBB.Engine.course as Course

# Reload libraries and modules
import importlib
importlib.reload(Utilities)
importlib.reload(Config)
importlib.reload(Course)

# Course names
names = ["The Last Black Box", "Build a Brain", "Bootcamp"]

# Build courses
for name in names:
    # Set paths
    if name == "The Last Black Box":
        Config.image_prefix = "../.."
    else:
        Config.image_prefix = "../../../.."

    # Load course
    print(f"Loading \"{name}\"...")
    course = Course.Course(name)

    # Render README for each session
    for session in course.sessions:
        README_text = session.render(name, type="MD")
        if name == "The Last Black Box":
            session_folder = f"{Config.boxes_root}/{session.slug}"
        else:
            session_folder = f"{Config.course_root}/versions/{course.slug}/{(session.index+1):02d}_{session.slug}"
        README_path = f"{session_folder}/README.md"
        with open(README_path, 'w', encoding='utf8') as f:
            f.writelines(README_text)    

# FIN