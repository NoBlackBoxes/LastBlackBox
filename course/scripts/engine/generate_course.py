# -*- coding: utf-8 -*-
"""
Generate LBB Courses

@author: kampff
"""

# Import Libraries
import os
import glob
import LBB.utilities as Utilities

# Import modules
import LBB.config as Config
import LBB.course as Course

# Reload libraries and modules
import importlib
importlib.reload(Utilities)
importlib.reload(Config)
importlib.reload(Course)

# Build course
name = "The Last Black Box"
# name = "Build a Brain"
# name = "Bootcamp"
course = Course.Course(name)

# Report
for session in course.sessions:
    print(session.name)

# FIN