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

# Build course
#name = "The Last Black Box"
name = "Build a Brain"
# name = "Bootcamp"
course = Course.Course(name)

# Report
for session in course.sessions:
    print(session.name)
    for box in session.boxes:
        print(f"- {box.name} : {box.depth}")
        for material in box.materials:
            print(f"  - {material.part}")

# FIN