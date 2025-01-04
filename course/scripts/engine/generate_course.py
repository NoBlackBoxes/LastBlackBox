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
#name = "Build a Brain"
name = "Bootcamp"
course = Course.Course(name)

# Debug
sessions = course.sessions[0:1]

# Print README for each sessions
for session in sessions:
    print(f"# {course.name} : {session.name}")
    print(f"{session.description}\n")
    for box in session.boxes:
        print(f"## {box.name} : {box.depth}")
        print(f"{box.description}\n")
        for material in box.materials:
            #print(f"  - {material.part}")
            pass
        for lesson in box.lessons:
            print(f"### {lesson.name}")
        print(f"\n")

# FIN